# ======================================================================================
# Linear Mixed Model to compare topological markers of different networks types 
# ======================================================================================
# Load necessary libraries
if (!require("pacman")) install.packages("pacman")
pacman::p_load(jsonlite, dplyr, tidyr, lme4, lmerTest, multcomp, emmeans, ggplot2)

# Define global variables, dicts, etc. 
topMarker_list <- list('clustering', 'modularity', 'participation')

densities <- c('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0')

folder <- "C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels/__topologicalMarker_pValue_lists"

modelsets <- c('topologicalMarker_dict_beRNN_highDim_', 
               'topologicalMarker_dict_beRNN_highDim_correctOnly_',
               'topologicalMarker_dict_brain_')

# Create a container to store p-values for plotting
p_value_results <- data.frame()


# ======================================================================================
# Q1: Main Loop
# ======================================================================================
for (d_idx in seq_along(densities)) {
  density <- densities[d_idx]
  basename <- paste0(modelsets[1], density)
  file_path <- file.path(folder, paste0(basename, ".json"))
  
  if (!file.exists(file_path)) next
  
  # ... [Keep your data parsing logic (data_dict -> df_set)] ...
  data_dict <- fromJSON(file_path)
  rows <- list()
  for (p_id in names(data_dict)) {
    markers <- data_dict[[p_id]]
    for (marker_raw in names(markers)) {
      if (marker_raw %in% topMarker_list) {
        values <- markers[[marker_raw]][1:5]
        for (m_idx in seq_along(values)) {
          rows[[length(rows) + 1]] <- data.frame(
            Participant = p_id,
            Model_ID = paste0(p_id, "_", m_idx),
            Marker_Type = marker_raw,
            Marker_Value = values[m_idx]
          )
        }
      }
    }
  }
  df_set <- bind_rows(rows)
  
  # Modeling
  fullModel <- lmer(Marker_Value ~ Marker_Type + (1 | Participant), data = df_set)
  
  # Extract p-value
  random_effects_test <- ranova(fullModel)
  p_val <- random_effects_test["(1 | Participant)", "Pr(>Chisq)"]
  
  # 3. Store result for plotting
  p_value_results <- rbind(p_value_results, data.frame(
    Density = as.numeric(density),
    P_Value = p_val
  ))
  
  print(paste("q1 p value:", p_val, "for density:", density))
}


# ======================================================================================
# Plotting the p-values across Densities (Consistent with Q2)
# ======================================================================================
# Define shared deterministic scale parameters
deterministic_ylim   <- c(1e-10, 1)
deterministic_breaks <- c(1, 0.05, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20, 1e-22, 1e-24, 1e-26, 1e-28, 1e-30)
deterministic_labels <- c("1", "0.05", "1e-4", "1e-6", "1e-8", "1e-10", "1e-12", "1e-14", "1e-16", "1e-18", "1e-20", "1e-22", "1e-24", "1e-26", "1e-28", "1e-30")
  
p_plot_q1 <- ggplot(p_value_results, aes(x = Density, y = P_Value)) +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red", alpha = 0.6) +
  geom_line(group = 1, color = "black", size = 1.2) +
  geom_point(color = "black", size = 3.5) +
  
  # Deterministic Log Scale
  scale_y_log10(
    breaks = deterministic_breaks,
    labels = deterministic_labels,
    expand = expansion(mult = c(0, 0.1)) # 10% breathing room at the top
  ) + 
  scale_x_continuous(breaks = seq(0.1, 1.0, by = 0.1)) +
  
  # Force the view to exactly 10^-6 to 1
  coord_cartesian(ylim = deterministic_ylim) +
  
  labs(
    title = "Significance of Participant Effect",
    subtitle = "LRT (ranova) for (1 | Participant)",
    x = "Network Density",
    y = "p-value (Log Scale)"
  ) +
  theme_bw() + 
  theme(
    panel.border = element_rect(colour = "black", fill = NA, size = 1.2),
    aspect.ratio = 0.75,
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 12)
  ) +
  annotate("text", x = 0.1, y = 0.58, label = "p = 0.05", 
           color = "red", hjust = 0, fontface = "italic")

ggsave(file.path(folder, "Q1_Participant_Effect_highDim.png"), p_plot_q1, width = 6, height = 3.5, dpi = 300)


# ======================================================================================
# Main Loop over Densities
# ======================================================================================
# Results storage
q2_results <- data.frame()

for (d_idx in seq_along(densities)) {
  density <- densities[d_idx]
  
  # File paths for comparison (Set 1 vs Set 3)
  path_1 <- file.path(folder, paste0(modelsets[2], density, ".json")) # beRNN
  path_3 <- file.path(folder, paste0(modelsets[2], density, ".json")) # beRNN or brain
  
  if (!file.exists(path_1) || !file.exists(path_3)) {
    message(paste("Skipping density", density, "- files missing."))
    next
  }
  
  # Helper function to parse JSON into a 'long' df
  parse_json_to_df <- function(file_path, group_label, set_id) {
    data <- fromJSON(file_path)
    rows <- list()
    for (p_id in names(data)) {
      for (marker_raw in names(data[[p_id]])) {
        if (marker_raw %in% topMarker_list) {
          values <- data[[p_id]][[marker_raw]][1:5]
          for (m_idx in seq_along(values)) {
            rows[[length(rows) + 1]] <- data.frame(
              Participant = p_id,
              Group = group_label,
              # Unique encoding for Model_ID is critical for repeated measures logic
              Model_ID = paste0(p_id, "_", set_id, "_", m_idx),
              Marker_Type = marker_raw,
              Marker_Value = values[m_idx]
            )
          }
        }
      }
    }
    return(bind_rows(rows))
  }
  
  # Load both sets
  df_set1 <- parse_json_to_df(path_1, "beRNN", "S1")
  df_set3 <- parse_json_to_df(path_3, "Brain", "S3")
  
  # Merge into combined dataframe
  df_all <- bind_rows(df_set1, df_set3) %>%
    mutate(across(c(Participant, Group, Marker_Type), as.factor))
  
  # ------------------------------------------------------------------
  # Q2 Model: Compare Datasets (Group) across Marker Types
  # ------------------------------------------------------------------
  # interaction Group*Marker_Type tests if dataset differences vary by marker
  m_q2 <- lmer(Marker_Value ~ Group * Marker_Type + (1 | Participant), data = df_all)
  
  # Extract p-value for the interaction effect
  anova_res <- anova(m_q2)
  p_interaction <- anova_res["Group:Marker_Type", "Pr(>F)"]
  
  # Run Post-hoc to see exactly where datasets differ
  posthoc <- emmeans(m_q2, pairwise ~ Group | Marker_Type)
  
  q2_results <- rbind(q2_results, data.frame(
    Density = as.numeric(density),
    P_Value = p_interaction
  ))
  
  print(paste("Processed density", density, "- Interaction P:", round(p_interaction, 5)))
}


# ======================================================================================
# Q2 Plot: Dataset Comparison (beRNN vs Brain)
# ======================================================================================

actual_min_p <- 1e-100
y_floor_q2 <- 10^floor(log10(actual_min_p)) 

# define y-ticks
# very small values > 20-steps (10^0, 10^-20, ...) best readable
shared_y_breaks_q2 <- 10^seq(1e-20, floor(log10(y_floor_q2)), by = -10)
shared_y_breaks_q2 <- sort(unique(c(shared_y_breaks_q2)))

p_q2_plot <- ggplot(q2_results, aes(x = Density, y = P_Value)) +
  # red significance line
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red", alpha = 0.6) +
  
  geom_line(color = "darkorange", size = 1.2) +
  geom_point(color = "darkorange", size = 3.5) +
  
  # log-scale
  scale_y_log10(
    breaks = shared_y_breaks_q2,
    labels = scales::label_log(),
    expand = expansion(mult = c(0.1, 0.1)) 
  ) +
  
  scale_x_continuous(breaks = seq(0.1, 1.0, by = 0.1)) +
  
  # define visual space
  coord_cartesian(ylim = c(y_floor_q2, 1)) + 
  
  labs(
    title = "Dataset Comparison: beRNN vs Brain",
    subtitle = "Interaction Significance (Group * Marker_Type)",
    x = "Network Density",
    y = "p-value (Log Scale)"
  ) +
  
  theme_bw() + 
  theme(
    aspect.ratio = 0.75, 
    panel.border = element_rect(colour = "black", fill = NA, size = 1.2),
    panel.grid.minor = element_blank(),
    axis.title = element_text(face = "bold"),
    plot.title = element_text(size = 12, face = "bold")
  ) +

  annotate("text", x = 0.1, y = 50, label = "p = 0.05", 
           color = "red", fontface = "italic", hjust = 0)

print(p_q2_plot)


# ======================================================================================
# save Q2 plots
# ======================================================================================
output_filename_q2 <- paste0("Q2_Comparison_beRNN_highDimCorrects_beRNN_highDimCorrects.png")
output_path_q2 <- file.path(folder, output_filename_q2)

ggsave(
  filename = output_path_q2,
  plot = p_q2_plot,
  width = 6,        
  height = 3.5,     
  dpi = 300,        
  bg = "white"      
)








