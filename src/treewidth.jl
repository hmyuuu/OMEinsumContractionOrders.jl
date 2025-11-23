"""
    struct Treewidth{EL <: EliminationAlgorithm} <: CodeOptimizer
    Treewidth(; alg::EL = SafeRules(BT(), MMW{3}(), MF()))

Tree-width based solver for contraction-order optimization. The heavy lifting is delegated to
[CliqueTrees.jl](https://algebraicjulia.github.io/CliqueTrees.jl/stable/) and
[TreeWidthSolver.jl](https://github.com/ArrogantGao/TreeWidthSolver.jl). `Treewidth` plugs an
elimination algorithm into the standard [`CodeOptimizer`](@ref) interface: it builds the line graph
of the einsum expression, computes a tree decomposition using `alg`, and turns that decomposition
into a contraction tree that can be evaluated by OMEinsum.

The default constructor uses the safe-rule pipeline `SafeRules(BT(), MMW{3}(), MF())`, combining
cheap heuristics with the exact Bouchitté–Todinca (`BT`) algorithm to obtain high-quality trees
without paying the exponential cost of `BT` on every instance.

# Fields
- `alg::EL`: elimination algorithm used to produce the tree decomposition. Any elimination algorithm
  exported by CliqueTrees.jl (e.g. `AMF`, `MF`, `MMD`, `LexBFS`, `LexM`, `BFS`, `MCS`, `RCMMD`,
  `RCMGL`, `MCSM`, `SafeRules`, `BT`, …) can be supplied.

# Related keywords
[`optimize_treewidth`](@ref) accepts the keyword `binary`. Set it to `false` to keep the multi-way
contraction tree delivered by the tree decomposition instead of binarizing it with an additional
`optimize_greedy_log2size` pass. (`binary=true` is usually preferred because it produces trees that
can be fed into BLAS-backed contractions directly.)

The elimination algorithms shipped with CliqueTrees span a range of cost/quality compromises:

| Algorithm | Description | Time Complexity | Space Complexity |
|:-----------|:-------------|:----------------|:-----------------|
| `AMF` | approximate minimum fill | O(m n) | O(m + n) |
| `MF` | minimum fill | O(m n²) | - |
| `MMD` | multiple minimum degree | O(m n²) | O(m + n) |
| `BT` | exact Bouchitté–Todinca solver | O(|Π| m n) | exponential in the treewidth |

where _n_ is the number of vertices in the line graph, _m_ is the number of edges and |Π| is the
number of potential maximal cliques. See the CliqueTrees documentation for the full catalogue of
available elimination algorithms.

# Example
```jldoctest
julia> optimizer = Treewidth();

julia> eincode = OMEinsumContractionOrders.EinCode([['a', 'b'], ['a', 'c', 'd'], ['b', 'c', 'e', 'f'], ['e'], ['d', 'f']], ['a'])
ab, acd, bcef, e, df -> a

julia> size_dict = Dict([c=>(1<<i) for (i,c) in enumerate(['a', 'b', 'c', 'd', 'e', 'f'])]...)
Dict{Char, Int64} with 6 entries:
  'f' => 64
  'a' => 2
  'c' => 8
  'd' => 16
  'e' => 32
  'b' => 4

julia> optcode = optimize_code(eincode, size_dict, optimizer)
ab, ba -> a
├─ ab
└─ bcf, acf -> ba
   ├─ bcef, e -> bcf
   │  ├─ bcef
   │  └─ e
   └─ acd, df -> acf
      ├─ acd
      └─ df
```
"""
Base.@kwdef struct Treewidth{EL <: EliminationAlgorithm} <: CodeOptimizer 
    alg::EL = SafeRules(BT(), MMW{3}(), MF())
end

"""
    const ExactTreewidth = Treewidth{SafeRules{BT, MMW{3}, MF}}
    ExactTreewidth() = Treewidth()

`ExactTreewidth` is a convenience alias for [`Treewidth`](@ref) configured with the safe-rule
pipeline that ends with the exact Bouchitté–Todinca (`BT`) solver from
[`TreeWidthSolver.jl`](https://github.com/ArrogantGao/TreeWidthSolver.jl). When it finishes, the
returned contraction tree has the minimal treewidth (and therefore the optimal time complexity)
of the input line graph.

The BT algorithm enumerates minimal separators and potential maximal cliques; its complexity is
`O(|Π| m n)` where |Π| is the number of potential maximal cliques, so the runtime is effectively
exponential in the treewidth of the instance. In practice `ExactTreewidth()` is ideal for networks
with a few dozen tensors and is commonly used to validate heuristic optimizers or generate golden
solutions for tests and benchmarks.
"""
const ExactTreewidth = Treewidth{SafeRules{BT, MMW{3}, MF}}
ExactTreewidth() = Treewidth()

"""
    optimize_treewidth(optimizer::Treewidth, code::AbstractEinsum, size_dict; binary=true)
    optimize_treewidth(optimizer::Treewidth, ixs, iy, size_dict; binary=true)

Compute a contraction tree by solving (exactly or approximately, depending on `optimizer.alg`)
the treewidth of the line graph associated with the einsum expression. The first method accepts
any [`AbstractEinsum`](@ref) object; the second operates directly on the list of input indices `ixs`,
the output indices `iy`, and the dictionary of index dimensions `size_dict`.

# Arguments
- `code` / (`ixs`, `iy`): representation of the einsum expression to optimize.
- `size_dict`: dictionary that maps each index label to its dimension. Internally it is converted to
  log₂ weights so CliqueTrees can work with weighted vertices.

Both methods return a [`NestedEinsum`](@ref) that can later be evaluated or further optimized.

# Keyword Arguments
- `binary`: if `true` (default), post-process the contraction tree with
  [`optimize_greedy_log2size`](@ref) to obtain a binary tree suitable for BLAS-backed contractions.
  Set this to `false` to receive the raw tree implied by the decomposition, which can be useful when
  experimenting with custom binarization strategies.
"""
function optimize_treewidth(optimizer::Treewidth, code::AbstractEinsum, size_dict::Dict; binary::Bool=true)
    optimize_treewidth(optimizer, getixsv(code), getiyv(code), size_dict; binary)
end

function optimize_treewidth(optimizer::Treewidth, ixs::AbstractVector{<:AbstractVector}, iy::AbstractVector, size_dict::Dict{L, Int}; binary::Bool=true) where {L}
    log2_size_dict = _log2_size_dict(size_dict)
    marker = zeros(Int, max(length(ixs) + 1, length(size_dict)))

    # construct incidence matrix `ve`
    #           indices
    #         [         ]
    # tensors [    ve   ]
    #         [         ]
    # we only care about the sparsity pattern
    weights, ev, ve, el = einexpr_to_matrix!(marker, ixs, iy, size_dict)

    # compute a tree (forest) decomposition of `ve`
    tree = matrix_to_tree!(marker, weights, ev, ve, el, optimizer.alg)

    # transform tree decomposition in contraction tree
    code = tree_to_einexpr!(marker, tree, ve, el, ixs, iy)

    if binary
        # binarize contraction tree
        code = optimize_greedy_log2size(code, log2_size_dict; α = 0.0, temperature = 0.0)
    end

    return code
end

"""
    einexpr_to_matrix!(marker, ixs, iy, size_dict)

Construct the weighted incidence matrix correponding to an Einstein summation expression.
Returns a quadruple (weights, ev, ve, el).

Each Einstein summation expression has a set E ⊆ L of indices, a set V := {1, …, |V|} of
(inner) tensors, and an outer tensor * := |V| + 1. Each tensor v ∈ V is incident to a sequence
ixs[v] of indices, and the outer tensor is incident to the sequence iy. Note that an index
can appear multiple times in ixs[v], e.g.

    ixs[v] = ('a', 'a', 'b').

Each index l ∈ E also has a positive dimension, given by size_dict[l].

The function `einexpr_to_matrix` does two things. First of all, it enumerates the index set
E, mapping each index to a distinct natural number.

    el: {1, …, |E|} → E
    le: E → {1, …, |E|}

Next, it constructs a vector weights: {1, …, |E|} → [0, ∞) satisfying

    weights[e] := log2(size_dict[el[e]]),

and a sparse matrix ve: {1, …, |V| + 1} × {1, …, |E|} → {0, 1} satisfying

    ve[v, e] := { 1 if el[e] is incident to v
                { 0 otherwise

We can think of the pair H := (weights, ve) as an edge-weighted hypergraph
with incidence matrix ve.
"""
function einexpr_to_matrix!(marker::AbstractVector{Int}, ixs::AbstractVector{<:AbstractVector{L}}, iy::AbstractVector{L}, size_dict::AbstractDict{L}) where {L}
    m = length(size_dict)
    n = length(ixs) + 1
    o = sum(length, ixs) + length(iy)

    # construct incidence matrix `ve`
    #           indices
    #         [         ]
    # tensors [    ve   ]
    #         [         ]
    # we only care about the sparsity pattern
    le = sizehint!(Dict{L, Int}(), m); el = sizehint!(L[], m) # el ∘ le = id
    weights = sizehint!(Float64[], m)
    colptr = sizehint!(Int[1], n + 1)
    rowval = sizehint!(Int[], o)
    nzval = sizehint!(Int[], o)

    # for all tensors v...
    for (v, ix) in enumerate(ixs)
        # for each index l incident to v...
        for l in ix
            # let e := le[l]
            if haskey(le, l)
                e = le[l]
            else
                push!(weights, log2(size_dict[l]))
                push!(el, l)
                e = le[l] = length(el)
            end

            # if l has not been seen before in ixs[v]...
            if marker[e] < v
                # mark e as seen
                marker[e] = v

                # set ev[e, v] := 1
                push!(rowval, e)
                push!(nzval, 1)
            end           
        end

        push!(colptr, length(rowval) + 1)
    end

    # v is the outer tensor
    v = length(colptr)

    # for each index l incident to v...
    for l in iy
        # let e := le[l]
        if haskey(le, l)
            e = le[l]
        else
            push!(weights, log2(size_dict[l]))
            push!(el, l)
            e = le[l] = length(el)
        end

        # if l has not been seen before in iy...
        if marker[e] < v
            # mark e as seen
            marker[e] = v

            # set ev[e, v] = 1
            push!(rowval, e)
            push!(nzval, 1)
        end           
    end

    push!(colptr, length(rowval) + 1)

    m = length(el)
    n = length(colptr) - 1

    ev = SparseMatrixCSC{Int, Int}(m, n, colptr, rowval, nzval)
    ve = copy(transpose(ev))
    return weights, ev, ve, el
end

"""
    matrix_to_tree!(marker, weights, ev, ve, el, alg)

Construct a tree decomposition of an edge-weighted hypergraph using
the elimination algorithm `alg`. We ensure that the indices incident
to the outer tensor are contained in the root bag of the tree decomposition.
"""
function matrix_to_tree!(marker::AbstractVector{Int}, weights::AbstractVector{Float64}, ev::SparseMatrixCSC{Int, Int}, ve::SparseMatrixCSC{Int, Int}, el::AbstractVector{L}, alg::EliminationAlgorithm) where {L}
    n, m = size(ve); tag = n + 1

    # construct line graph `ee`
    #           indices
    #         [         ]
    # indices [    ee   ]
    #         [         ]
    # we only care about the sparsity pattern
    ee = ve' * ve

    # compute a tree (forest) decomposition of ee
    perm, tree = cliquetree(weights, ee; alg=ConnectedComponents(alg))

    # find the bag containing iy, call it root
    root = length(tree)

    # mark the indices in iy
    for e in view(rowvals(ev), nzrange(ev, n))
        marker[e] = tag
    end

    # the first bag containing an index in iy
    # must contain all of iy
    for (b, bag) in enumerate(tree)
        root < length(tree) && break

        for e in residual(bag)
            root < length(tree) && break

            if marker[perm[e]] == tag
                root = b
            end
        end
    end

    # make root a root node of the tree decomposition
    permute!(perm, cliquetree!(tree, root))

    # permute incidence matrix `ve` and label vector `el`.
    permute!(ve, axes(ve, 1), perm)
    permute!(el, perm)

    return tree
end

"""
    tree_to_einexpr!(marker, tree, ve, el, ixs, iy)

Transform a tree decomposition into a contraction tree.
"""
function tree_to_einexpr!(marker::AbstractVector{Int}, tree::CliqueTree{Int, Int}, ve::SparseMatrixCSC{Int, Int}, el::AbstractVector{L}, ixs::AbstractVector{<:AbstractVector{L}}, iy::AbstractVector{L}) where {L}
    n, m = size(ve); tag = n + 2

    # dynamic programming
    stack = NestedEinsum{L}[]

    # for each bag b...
    for (b, bag) in enumerate(tree)
        # sep is the separator at b
        sep = separator(bag)

        # res is the residual at b
        res = residual(bag)

        # code is the Einstein summation expression at b
        code = NestedEinsum(NestedEinsum{L}[], EinCode(Vector{L}[], L[]))

        for e in sep
            push!(code.eins.iy, el[e])
        end

        # for each index e in the residual...
        for e in res
            # for each tensor v indicent to e...
            for v in view(rowvals(ve), nzrange(ve, e))
                # if has not been seen before...
                if marker[v] < tag
                    # mark v as seen
                    marker[v] = tag

                    # if v is the outer tensor...
                    if v == n
                        # expose iy
                        append!(code.eins.iy, iy)
                    # if v is an inner tensor...
                    else
                        # make v a child of code
                        push!(code.args, NestedEinsum{L}(v))
                        push!(code.eins.ixs, ixs[v])
                    end
                end
            end
        end

        # for each child bag of b...
        for _ in childindices(tree, b)
            # the Einstein summation expression corresponding to
            # the child bag is at the top of the stack
            child = pop!(stack)

            # make this expression a child of code
            push!(code.args, child)
            push!(code.eins.ixs, child.eins.iy)
        end

        # push code to the stack
        push!(stack, code)
    end

    # we now have an expression for each root of the tree decomposition.
    # merge these together into a single Einstein expression code.
    if isone(length(stack))
        code = only(stack)
    else
        code = NestedEinsum(NestedEinsum{L}[], EinCode(Vector{L}[], L[]))
        append!(code.eins.iy, iy)

        while !isempty(stack)
            child = pop!(stack)
            push!(code.args, child)
            push!(code.eins.ixs, child.eins.iy)
        end
    end

    # append scalars to code
    for (v, ix) in enumerate(ixs)
        if isempty(ix)
            push!(code.args, NestedEinsum{L}(v))
            push!(code.eins.ixs, ix)
        end
    end

    return code
end

# no longer used
function il2lg(incidence_list::IncidenceList{VT, ET}, indicies::Vector{ET}) where {VT, ET}
    line_graph = SimpleGraph(length(indicies))
    
    for (i, e) in enumerate(indicies)
        for v in incidence_list.e2v[e]
            for ej in incidence_list.v2e[v]
                if e != ej add_edge!(line_graph, i, findfirst(==(ej), indicies)) end
            end
        end
    end

    return line_graph
end
