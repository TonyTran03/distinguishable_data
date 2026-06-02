library(Rtsne)
library(igraph)

set.seed(2024)

script_arg <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", script_arg[grepl("^--file=", script_arg)])
if (length(script_file) == 0) {
  script_file <- "data_synthesis/src/tSNE_partial_correlation.R"
}
repo_root <- normalizePath(file.path(dirname(script_file[1]), "..", ".."))
input_dir <- file.path(repo_root, "data_synthesis", "notebooks", "revision_cache", "r_tsne_partial_correlation")
output_dir <- input_dir

partial_path <- file.path(input_dir, "hiv_real_glasso_partial_corr.csv")
metadata_path <- file.path(input_dir, "hiv_real_glasso_feature_metadata.csv")

partial_df <- read.csv(partial_path, row.names = 1, check.names = FALSE)
metadata <- read.csv(metadata_path, check.names = FALSE)

partial <- as.matrix(partial_df)
diag(partial) <- 0
partial[!is.finite(partial)] <- 0

profiles <- scale(partial)
profiles[!is.finite(profiles)] <- 0

perplexity <- 20
z <- Rtsne(
  profiles,
  check_duplicates = FALSE,
  dims = 2,
  perplexity = perplexity,
  pca = FALSE,
  max_iter = 10000,
  verbose = TRUE
)

layout_tsne <- z$Y
write.csv(
  data.frame(index = metadata$index, feature = metadata$feature, tSNE1 = layout_tsne[, 1], tSNE2 = layout_tsne[, 2]),
  file.path(output_dir, "hiv_real_glasso_partial_tsne_coords_R.csv"),
  row.names = FALSE
)

threshold <- 1e-7
adj <- abs(partial)
adj[adj <= threshold] <- 0
diag(adj) <- 0

g <- graph_from_adjacency_matrix(adj, mode = "undirected", weighted = TRUE, diag = FALSE)
E(g)$signed_weight <- partial[as_edgelist(g, names = FALSE)]
V(g)$label <- metadata$index

feature_colours <- c(
  rep("#DC3A7A", 8),
  rep("#9E3934", 10),
  rep("#ED220D", 10),
  rep("skyblue", 10),
  rep("#3675B8", 10),
  rep("#F8B959", 9),
  rep("#94C652", 6)
)
if (length(feature_colours) != vcount(g)) {
  feature_colours <- rep("#D83A3A", vcount(g))
}

strength <- metadata$strength
vertex_sizes <- 6 + 10 * sqrt(strength / max(strength, na.rm = TRUE))
edge_ramp <- colorRampPalette(c("#F0F0F0", "#484848"))

plot_network <- function(graph, filename, main, edge_width_scale = 2.0) {
  edge_weights <- E(graph)$weight
  if (length(edge_weights) == 0) {
    edge_colors <- "#D8D8D8"
    edge_widths <- 0.5
  } else {
    ranks <- rank(edge_weights, ties.method = "first")
    edge_colors <- edge_ramp(length(edge_weights))[ranks]
    edge_widths <- 0.25 + edge_width_scale * edge_weights / max(edge_weights)
  }

  png(file.path(output_dir, filename), width = 1500, height = 1200, res = 180)
  par(mar = c(2, 2, 4, 2))
  plot(
    graph,
    layout = layout_tsne,
    vertex.label = V(graph)$label,
    vertex.label.cex = 0.62,
    vertex.label.color = "#111111",
    vertex.shape = "circle",
    vertex.color = feature_colours,
    vertex.frame.color = feature_colours,
    vertex.size = vertex_sizes,
    edge.arrow.size = 0.01,
    edge.width = edge_widths,
    edge.color = edge_colors,
    main = main
  )
  dev.off()
}

plot_network(
  g,
  "hiv_real_glasso_partial_tsne_all_edges_R.png",
  paste0("HIV Graphical Lasso partial-correlation feature network\nRtsne perplexity = ", perplexity)
)

top_n <- min(80, ecount(g))
top_edges <- order(E(g)$weight, decreasing = TRUE)[seq_len(top_n)]
g_top <- subgraph_from_edges(g, eids = top_edges, delete.vertices = FALSE)
plot_network(
  g_top,
  "hiv_real_glasso_partial_tsne_top80_edges_R.png",
  paste0("HIV Graphical Lasso partial-correlation feature network\nTop ", top_n, " absolute partial-correlation edges")
)

g_mst <- g
E(g_mst)$distance <- 1 / pmax(E(g_mst)$weight, 1e-12)
g_mst <- mst(g_mst, weights = E(g_mst)$distance)
plot_network(
  g_mst,
  "hiv_real_glasso_partial_tsne_mst_backbone_R.png",
  "HIV Graphical Lasso partial-correlation feature network\nMST backbone on Rtsne layout",
  edge_width_scale = 3.0
)

save(z, g, g_top, g_mst, layout_tsne, partial, metadata, file = file.path(output_dir, "hiv_real_glasso_partial_tsne_R_outputs.RData"))

cat("Wrote R t-SNE partial-correlation outputs to:", output_dir, "\n")
