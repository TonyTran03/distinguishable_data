# Load required libraries
library(MASS)  # For generating synthetic data
library(Rtsne)  # For t-SNE
library(igraph)  # For network visualization
library(Hmisc)  # For correlation test
library(grDevices)

# Set seed for reproducibility
set.seed(123)

# Generate synthetic feature matrix (10 individuals, 20 features)
#num_individuals <- 10
#num_features <- 20
#synthetic_data <- mvrnorm(num_individuals, mu = rep(0, num_features), Sigma = diag(num_features))

features_all.imp = read.csv("features_withMissForestImputation_IR_INR_together_jul25_63Features.csv")
x = features_all.imp[1:91,2:64]
num_features <- length(x[1,])
num_individuals <- length(x[,1])
importance = read.csv("ImportanceFactor_downSampled_control_IRandINR.csv")
StatusFile = read.csv("Status_Mario_Brockman.csv", sep = ",", header = T)
load("underTheHood/FeatureWeights_allID_50Iterations.RData")

load("tSNE_layoutInformation.RData")

health_outcome = StatusFile$HIVStatus[1:91]
health_outcome[which(health_outcome == 2)] = 1
health_outcome[which(health_outcome == 3)] = 1
health_outcome[which(health_outcome == 4)] = 1
y = health_outcome

# Print the synthetic data

x <- t(x) #Take Transpose to run the SNE algorithm on features (versus individuals )
# Apply t-SNE algorithm
tsne_result <- Rtsne(x, dims = 2, perplexity = 5, verbose = TRUE, max_iter =10000)

# Extract t-SNE coordinates
tsne_coords <- tsne_result$Y
layout_tsne <- as.matrix(tsne_coords)
x = as.matrix(x)
# Print t-SNE coordinates
#print(tsne_coords)

# Compute correlation matrix
correlation_matrix <- cor(x)
cor_test_results <- rcorr(x)
correlation_matrix <- cor_test_results$r
p_value_matrix <- cor_test_results$P

min_p_value <- .Machine$double.eps  # Smallest positive value representable
p_value_matrix[p_value_matrix == 0] <- min_p_value

# Create a graph from the correlation matrix
# Here we create edges based on a threshold correlation value
threshold <- 0.5
adj_matrix <- abs(correlation_matrix) >= threshold

# Create the graph object
g <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected", diag = FALSE)
vertex_sizes <- degree(g)  

edge_list <- as_edgelist(g, names = FALSE)
# Compute edge weights based on -log10(p-values)
edge_weights <- sapply(1:nrow(edge_list), function(i) {
  v1 <- as.numeric(edge_list[i, 1])
  v2 <- as.numeric(edge_list[i, 2])
  -log10(p_value_matrix[v1, v2])
})


E(g)$weight <- edge_weights

#For Individuals
#olours = c(rep("firebrick", 23), rep("skyblue", 68))
colours = c(rep("firebrick", 23), rep("skyblue", 43), rep("chartreuse3", 25))

#For features:
colours = c(rep("#DC3A7A", 8), rep("#9E3934", 10),rep("#ED220D", 10), rep("skyblue", 10),rep("#3675B8", 10), rep("#F8B959", 9), rep("#94C652", 6))
#num_features <- length(x[1,])
#num_individuals <- length(x[,1])

# Define the start and end colors for edges
start_color <- "#FFFFFF"
end_color <- "#7F7F7F"

# Create the color ramp palette
color_ramp <- colorRampPalette(c(start_color, end_color))(length(edge_list[,2]))
#Make size of vertex proportional to their degree (number of connections)

quartz()
plot(g, layout = layout_tsne, 
     vertex.label = 1:num_individuals,
     #vertex.size = (features_all.imp$RATIO_CD4CD8*5),
     vertex.size = 15,
     #vertex.size = vertex_sizes,
     #vertex.size = E(g)$weight*1.5,
     vertex.color = colours,
     vertex.frame.color = colours,
     edge.color = color_ramp,
     main = "Correlation Individual Network Visualized with t-SNE perplexity = 5")

#Visualize edges as length proportional to -log10(p-values)


# # # # # # # Code from Sajjad # # # # # # # 

library(Rtsne)
library(Hmisc)
library(igraph)
set.seed(2024)

Data = x
Data = as.matrix(Data)
temp=rcorr(Data, type='spearman')
pvs2=temp$P
pvs2=pvs2+min(pvs2[pvs2!=0],na.rm=TRUE)
cors=temp$r
pvs2=-log10(pvs2)
pvs2[pvs2==0]=NA
pvs2[cors==1]=NA
pvs2[is.na(pvs2)]=0

g=graph_from_adjacency_matrix(pvs2, mode='undirected', weighted=TRUE)
edge_list <- as_edgelist(g, names = FALSE)
z=Rtsne(pvs2, check_duplicates = FALSE, 2, perplexity=((nrow(pvs2)-1)/3))
#plot(g, edge.width=.5, layout=z$Y, edge.arrow.size=0.01, vertex.shape='circle', vertex.frame.color=NA)

colours = c(rep("#DC3A7A", 8), rep("#9E3934", 10),rep("#ED220D", 10), rep("skyblue", 10),rep("#3675B8", 10), rep("#F8B959", 9), rep("#94C652", 6))
start_color <- "#F0F0F0"
end_color <- "#484848"
color_ramp <- colorRampPalette(c(start_color, end_color))(length(edge_list[,2]))

quartz()
#Plot all edges as gradient
#Node size is proportional to importance factor from IRandINR
#Node colour is relative to the feature type
plot(g, edge.width=.5, layout=z$Y, vertex.label = 1:63, edge.arrow.size=0.01, vertex.shape='circle', vertex.color = colours,
     vertex.frame.color = colours,vertex.size = sqrt(importance$x)*25, edge.color = color_ramp,
     main = "Correlation feature Network Visualized \n with t-SNE perplexity = (nrow(pvs2)-1)/3")

#Use colour based on minimum KLD from synthetic matrices
#From Control_IRandINR_syntheticData.R

minimum.allFeatures.byKLD
#colour code order: 
#x.GMM,
#x.MVN,
#x.SMOTE,
#x.k3,
#x.koptimal,
#x.k20

color_palette <- c("#FF0000",  # Red
                   "#00FF00",  # Green
                   "#0000FF",  # Blue
                   "#FFFF00",  # Yellow
                   "#FF00FF",  # Magenta
                   "#00FFFF")  # Cyan

color_palette <- c("#2E8B57",  # Sea Green
                   "#1E90FF",  # Dodger Blue
                   "#FFD700",  # Gold
                   "#FF6347",  # Tomato (Red-Orange)
                   "#8A2BE2",  # Blue Violet
                   "#FF69B4")  # Hot Pink
colours.syntheticByKLD <- color_palette[minimum.allFeatures.byKLD]


quartz()
#Plot all edges as gradient
#Node size is proportional to importance factor from IRandINR
#Node colour is relative to the feature type
plot(g, edge.width=.5, layout=z$Y, vertex.label = 1:63, edge.arrow.size=0.01, vertex.shape='circle', vertex.color =  colours.syntheticByKLD,
     vertex.frame.color = colours.syntheticByKLD,
     #vertex.size = sqrt(importance$x)*25,
     vertex.size = 8,
     edge.color = color_ramp,
     main = "Colour based on minimum KLD")

################################################################################
######### Plot only those edges with p-values less than 0.05 ########
################################################################################

library(Rtsne)
library(Hmisc)
library(igraph)
set.seed(2024)

# Your original data processing
Data = x
Data = as.matrix(Data)
Data = t(Data)
temp = rcorr(Data, type = 'spearman')
pvs2 = temp$P
pvs2 = pvs2 + min(pvs2[pvs2 != 0], na.rm = TRUE)  # Smallest non-zero p-value adjustment
cors = temp$r
pvs2 = -log10(pvs2)  # Convert p-values to -log10 scale
pvs2[pvs2 == 0] = NA  # Set 0 values to NA
pvs2[cors == 1] = NA  # Ignore self-correlation (correlation of 1)
pvs2[is.na(pvs2)] = 0  # Replace NA with 0

# Create the graph from the adjacency matrix
g = graph_from_adjacency_matrix(pvs2, mode = 'undirected', weighted = TRUE)
edge_list <- as_edgelist(g, names = FALSE)

# Filter edges where p-value is less than 0.05 (in log10 scale, this is -log10(0.05))
threshold <- -log10(0.05)
edges_to_keep <- which(E(g)$weight > threshold)

# Create a subgraph with only the filtered edges
g_filtered = subgraph.edges(g, eids = edges_to_keep, delete.vertices = FALSE)

# Perform t-SNE on the p-values matrix: this one is used in the paper (on data columns not individuals)
z = Rtsne(pvs2, check_duplicates = FALSE, dims = 2, perplexity = ((nrow(pvs2) - 1) / 3))

# Define colours for nodes and edges
colours = c(rep("#DC3A7A", 8), rep("#9E3934", 10), rep("#ED220D", 10), rep("skyblue", 10),
            rep("#3675B8", 10), rep("#F8B959", 9), rep("#94C652", 6))
start_color <- "#F0F0F0"
end_color <- "#484848"
color_ramp <- colorRampPalette(c(start_color, end_color))(length(E(g_filtered)))

# Plot the filtered graph
quartz()
plot(g_filtered, edge.width = 0.5, layout = z$Y, vertex.label = 1:63, 
     edge.arrow.size = 0.01, vertex.shape = 'circle', vertex.color = colours,
     vertex.frame.color = colours, vertex.size = sqrt(importance$x) * 25, edge.color = color_ramp,
     main = "Correlation Feature Network Visualized \n with t-SNE perplexity = 3")

quartz.save("tSNE_Perplexity3.png", type = "png")

quartz()
plot(g, edge.width = 0.5, layout = z$Y, vertex.label = 1:63, 
     edge.arrow.size = 0.01, vertex.shape = 'circle', vertex.color = colours,
     vertex.frame.color = colours, vertex.size = sqrt(importance$x) * 25, edge.color = color_ramp,
     main = "Correlation Feature Network Visualized \n with t-SNE perplexity = (nrow(pvs2) - 1) / 3")

####Plot subset of edges

# Define the vertices of interest
vertices_of_interest <- 49:57

# Find edges connected to vertices 49:57 and with p-value < 0.05
edges_connected_to_vertices <- E(g_filtered)[from(vertices_of_interest) | to(vertices_of_interest)]

# Create a new subgraph from these edges
g_subgraph <- subgraph.edges(g_filtered, eids = edges_connected_to_vertices, delete.vertices = FALSE)

# Plot the new subgraph with the same layout
quartz()
plot(g_subgraph, edge.width = 0.5, layout = z$Y, vertex.label = 1:63, 
     edge.arrow.size = 0.01, vertex.shape = 'circle', vertex.color = colours,
     vertex.frame.color = colours, vertex.size = sqrt(importance$x) * 25, edge.color = color_ramp,
     main = "Subgraph: Vertices 49 to 57 with p-value < 0.05")

start_color_edge <- "#BABABA"
end_color_edge <- "#484848"
color_ramp_edge <- colorRampPalette(c(start_color_edge, end_color_edge))(length(edge_colors[edges_to_plot]))
edge_colors[edges_to_plot] = color_ramp_edge

quartz()
plot(g_subgraph, edge.width = 0.5, layout = z$Y, vertex.label = 1:63, 
     edge.arrow.size = 0.01, vertex.shape = 'circle',
     edge.color = edge_colors,
     vertex.color = full_colors[64:126],
     vertex.frame.color = full_colors[64:126],
     vertex.size = sqrt(importance$x) * 25, 
     main = "Subgraph: Vertices 49 to 57 with p-value < 0.05, HIV-")

quartz()
plot(g_subgraph, edge.width = 0.5, layout = z$Y, vertex.label = 1:63, 
     edge.arrow.size = 0.01, vertex.shape = 'circle',
     edge.color = edge_colors,
     vertex.color = full_colors[1:63],
     vertex.frame.color = full_colors[1:63],
     vertex.size = sqrt(importance$x) * 25, 
     main = "Subgraph: Vertices 49 to 57 with p-value < 0.05, HIV+")






tsne_points = z$Y

V(g)$x <- tsne_points[, 1]
V(g)$y <- tsne_points[, 2]

subset_vertices <- 49:57

edges_to_plot <- E(g)[.inc(V(g)[subset_vertices])]


# Create a vector for edge colors
edge_colors <- rep(NA, ecount(g))
edge_colors[edges_to_plot] <- "black"

start_color_edge <- "#BABABA"
end_color_edge <- "#484848"
color_ramp_edge <- colorRampPalette(c(start_color_edge, end_color_edge))(length(edge_colors[edges_to_plot]))
edge_colors[edges_to_plot] = color_ramp_edge

threshold <- -log10(0.05)
edges_to_keep <- which(E(g)$weight > threshold)

# Create a subgraph with only the filtered edges
g_filtered_2 = subgraph.edges(g, eids = edges_to_keep, delete.vertices = FALSE)


quartz()
plot(
  g,
  layout = z$Y,
  edge.color = edge_colors,
  vertex.color = full_colors[64:126],
  vertex.frame.color = full_colors[64:126],
  vertex.size = sqrt(importance$x)*25,
  vertex.label = 1:63,
  edge.width = 1,
  #edge.curved = 0.2,
  main = "t-SNE Plot with Subset of Edges"
)

quartz()
plot(
  g,
  layout = z$Y,
  edge.color = edge_colors,
  vertex.color = full_colors[1:63],
  vertex.frame.color = full_colors[1:63],
  vertex.size = sqrt(importance$x)*25,
  vertex.label = 1:63,
  edge.width = 1,
  #edge.curved = 0.2,
  main = "t-SNE Plot with Subset of Edges"
)





################################################################################
################################################################################

#From the under the hood script: 
load("underTheHood/FeatureWeights_allID_50Iterations.RData")
#From the above, "my_Array" holds all the feature weights
#Seq of HIV- who is always guessed correctly
seqHIVNEG_correct = c(1, 2, 4, 5, 9, 13, 14 , 15, 16 , 17, 18, 19, 20, 21, 22, 23)
#Seq of HIV- who tend to be guessed incorrectly
seqHIVNEG_INcorrect = c(6, 7, 11, 12)
#Seq of HIV+ who is always guessed correctly 
seqHIVPOS_correct = c(25, 26, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 41, 42, 43, 44, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 65, 66, 68, 70, 71, 73, 74, 78, 79, 80, 81, 82, 83, 84, 85, 86, 90, 91)
#Seq of HIV+ who is usually guessed incorrectly 
seqHIVPOS_INcorrect = c(39, 40, 51, 67, 72, 75, 77, 87)
#my_array[id, feature, iteration]
IDsequence = seqHIVNEG_correct


load("tSNE_layoutInformation.RData")
#Need to put feature weight on a gradient from red to blue: red is max negative through to blue which is max positive.
save(z, g, edge_list, file = "tSNE_layoutInformation.RData")

#We want the node colours to correspond to the feature weight medians. 
storeMedians = numeric(63)
storeMedians_HIVpos_correct = numeric(63)
storeMedians_HIVNEG_correct = numeric(63)
for(i in 1:63){
  storeMedians_HIVpos_correct[i] = median(my_array[seqHIVPOS_correct,i,1:50])
  storeMedians_HIVNEG_correct[i] = median(my_array[seqHIVNEG_correct,i,1:50])
  #Use the min(HUVneg correct) and max(HIVpos_correct) range of medians of medians to define the colour gradient. 
}

# Define the gradient range
gradient_min <- min(storeMedians_HIVNEG_correct)  # Define minimum value of the gradient
gradient_max <- max(storeMedians_HIVpos_correct)   # Define maximum value of the gradient


# Normalization function
normalize <- function(x, min_value, max_value) {
  normalized <- (x - min_value) #/ (max_value - min_value)
  #normalized[normalized < 0] <- 0  # Clamp values below 0
  #normalized[normalized > 1] <- 1  # Clamp values above 1
  return(normalized)
}

start_color_nodes <- "#2E5F78"
middle_color_nodes <- "#E9F0F6"
end_color_nodes <- "#9A3F23"

color_gradient <- colorRampPalette(c(start_color_nodes, middle_color_nodes , end_color_nodes))(126)
fullVector = c(storeMedians_HIVpos_correct, storeMedians_HIVNEG_correct)
normalized_full_vector <- normalize(fullVector, gradient_min, gradient_max)
full_colors <- color_gradient[as.numeric(cut(normalized_full_vector, breaks = 126))]

#normalized_values <- (storeMedians - min(storeMedians)) / (max(storeMedians) - min(storeMedians))
#colors <- gradient_color(length(storeMedians))[as.numeric(cut(normalized_values, breaks = length(storeMedians)))]
quartz()
plot(fullVector, col = full_colors, pch = 19, cex = 1.5,
     main = "Point Plot with Gradient Colors",
     xlab = "Index", ylab = "Value", ylim = c(-0.12, 0.13))


start_color_edge <- "#F0F0F0"
end_color_edge <- "#484848"
color_ramp_edge <- colorRampPalette(c(start_color_edge, end_color_edge))(length(edge_list[,2]))
quartz()

plot(g, edge.width=.5, layout=z$Y, vertex.label = 1:63, edge.arrow.size=0.01, vertex.shape='circle', vertex.color = full_colors[64:126],
     vertex.frame.color = full_colors[64:126],vertex.size = sqrt(importance$x)*25, edge.color = color_ramp,
     main = "Correlation feature Network Visualized \n with t-SNE perplexity = (nrow(pvs2)-1)/3")

#Okay, next I want to only plot a subet of edges that correspond to the Cytokine Features
g=graph_from_adjacency_matrix(pvs2, mode='undirected', weighted=TRUE)
edge_list <- as_edgelist(g, names = FALSE)
z=Rtsne(pvs2, check_duplicates = FALSE, 2, perplexity=((nrow(pvs2)-1)/3))

tsne_points = z$Y

V(g)$x <- tsne_points[, 1]
V(g)$y <- tsne_points[, 2]

subset_vertices <- 49:57

edges_to_plot <- E(g)[.inc(V(g)[subset_vertices])]

# Create a vector for edge colors
edge_colors <- rep(NA, ecount(g))
edge_colors[edges_to_plot] <- "black"

start_color_edge <- "#F0F0F0"
end_color_edge <- "#BABABA"
color_ramp_edge <- colorRampPalette(c(start_color_edge, end_color_edge))(length(edge_colors[edges_to_plot]))
edge_colors[edges_to_plot] = color_ramp_edge

threshold <- -log10(0.05)
edges_to_keep <- which(E(g)$weight > threshold)

# Create a subgraph with only the filtered edges
g_filtered_2 = subgraph.edges(g, eids = edges_to_keep, delete.vertices = FALSE)


quartz()
plot(
  g,
  layout = z$Y,
  edge.color = edge_colors,
  vertex.color = full_colors[64:126],
  vertex.frame.color = full_colors[64:126],
  vertex.size = sqrt(importance$x)*25,
  vertex.label = 1:63,
  edge.width = 1,
  #edge.curved = 0.2,
  main = "t-SNE Plot with Subset of Edges"
)

quartz()
plot(
  g,
  layout = z$Y,
  edge.color = edge_colors,
  vertex.color = full_colors[1:63],
  vertex.frame.color = full_colors[1:63],
  vertex.size = sqrt(importance$x)*25,
  vertex.label = 1:63,
  edge.width = 1,
  #edge.curved = 0.2,
  main = "t-SNE Plot with Subset of Edges"
)
g_filtered

quartz()
plot(
  g_filtered_2,
  layout = z$Y,
  edge.color = edge_colors,
  vertex.color = full_colors[64:126],
  vertex.frame.color = full_colors[64:126],
  vertex.size = sqrt(importance$x)*25,
  vertex.label = 1:63,
  edge.width = 1,
  #edge.curved = 0.2,
  main = "t-SNE Plot with Subset of Edges"
)
#Can I keep the g and layout, but recompute the edges ??? Number of features would remain the same...
#Okay, next I want to only plot a subet of edges that correspond to the Cytokine Features

#Seq of HIV- who is always guessed correctly
seqHIVNEG_correct = c(1, 2, 4, 5, 9, 13, 14 , 15, 16 , 17, 18, 19, 20, 21, 22, 23)
#Seq of HIV- who tend to be guessed incorrectly
seqHIVNEG_INcorrect = c(6, 7, 11, 12)
#Seq of HIV+ who is always guessed correctly 
seqHIVPOS_correct = c(25, 26, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 41, 42, 43, 44, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 65, 66, 68, 70, 71, 73, 74, 78, 79, 80, 81, 82, 83, 84, 85, 86, 90, 91)
#Seq of HIV+ who is usually guessed incorrectly 
seqHIVPOS_INcorrect = c(39, 40, 51, 67, 72, 75, 77, 87)
#my_array[id, feature, iteration]
IDsequence = c(seqHIVNEG_correct,seqHIVNEG_INcorrect)

Data2 = x[IDsequence,]
Data2 = as.matrix(Data2)
temp2=rcorr(Data2, type='spearman')
pvs2=temp2$P
pvs2=pvs2+min(pvs2[pvs2!=0],na.rm=TRUE)
cors=temp2$r
pvs2=-log10(pvs2)
pvs2[pvs2==0]=NA
pvs2[cors==1]=NA
pvs2[is.na(pvs2)]=0


g2=graph_from_adjacency_matrix(pvs2, mode='undirected', weighted=TRUE)
edge_list2 <- as_edgelist(g2, names = FALSE)
z2=Rtsne(pvs2, check_duplicates = FALSE, 2, perplexity=((nrow(pvs2)-1)/3))

tsne_points2 = z2$Y

V(g2)$x <- tsne_points2[, 1]
V(g2)$y <- tsne_points2[, 2]

subset_vertices <- 49:57

edges_to_plot2 <- E(g2)[.inc(V(g2)[subset_vertices])]

# Create a vector for edge colors
edge_colors2 <- rep(NA, ecount(g2))
edge_colors2[edges_to_plot2] <- "black"

start_color_edge <- "#F0F0F0"
end_color_edge <- "#BABABA"
color_ramp_edge2 <- colorRampPalette(c(start_color_edge, end_color_edge))(length(edge_colors2[edges_to_plot2]))
edge_colors2[edges_to_plot2] = color_ramp_edge2

quartz()
plot(
  g,
  layout = z$Y,
  edge.color = edge_colors,
  vertex.color = full_colors[64:126],
  vertex.frame.color = full_colors[64:126],
  vertex.size = sqrt(importance$x)*25,
  vertex.label = 1:63,
  edge.width = 0.8,
  #edge.curved = 0.2,
  main = "t-SNE Plot with Subset of Edges"
)

quartz()
plot(
  g2,
  layout = z2$Y,
  edge.color = edge_colors2,
  vertex.color = full_colors[64:126],
  vertex.frame.color = full_colors[64:126],
  vertex.size = sqrt(importance$x)*25,
  vertex.label = 1:63,
  edge.width = 0.8,
  #edge.curved = 0.2,
  main = "t-SNE Plot with Subset of Edges"
)
