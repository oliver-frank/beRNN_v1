# ======================================================================================
# Linear Mixed Model to compare topological markers of different networks types 
# ======================================================================================
# Load necessary libraries
if (!require("pacman")) install.packages("pacman")
pacman::p_load(jsonlite, dplyr, tidyr, lme4, lmerTest, multcomp, emmeans, ggplot2, performance)


# Define global variables, dicts, etc. 
topMarker_list <- list('clustering', 'modularity', 'participation')

densities <- c('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0')

folder <- "C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels/__topologicalMarker_pValue_lists"

modelsets <- c('topologicalMarker_dict_beRNN_highDim_', 
               'topologicalMarker_dict_beRNN_highDim_correctOnly_',
               'topologicalMarker_dict_brain_')

# Create a container to store p-values for plotting
p_value_results <- data.frame()

use_averaging <- FALSE





# ======================================================================================
# Main Loop over Densities
# ======================================================================================
# Results storage
q2_results <- data.frame()
posthoc_results <- data.frame()

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
          
          # Wir nehmen die ersten 5 Werte
          values <- data[[p_id]][[marker_raw]][1:5]
          
          # --- AGGREGATION LOGIC ---
          if (use_averaging) {
            # Berechne den Mittelwert der 5 Modelle pro Proband & Marker
            rows[[length(rows) + 1]] <- data.frame(
              Participant = p_id,
              Group = group_label,
              Pair_ID = "Avg", # Alle erhalten die gleiche ID
              Marker_Type = marker_raw,
              Marker_Value = mean(values, na.rm = TRUE)
            )
          } else {
            # ORIGINAL: Alle 5 Modelle einzeln behalten
            for (m_idx in seq_along(values)) {
              rows[[length(rows) + 1]] <- data.frame(
                Participant = p_id,
                Group = group_label,
                Pair_ID = as.factor(m_idx), 
                Marker_Type = marker_raw,
                Marker_Value = values[m_idx]
              )
            }
          }
          # -------------------------
        }
      }
    }
    return(bind_rows(rows))
  }
  
  # Load both sets
  df_set1 <- parse_json_to_df(path_1, "beRNN_corr.", "S2")
  df_set3 <- parse_json_to_df(path_3, "brain.", "S3")
  
  # Merge into combined dataframe
  df_all <- bind_rows(df_set1, df_set3) %>%
    mutate(across(c(Participant, Group, Marker_Type, Pair_ID), as.factor))
  
  
  
  # ------------------------------------------------------------------
  # Q2 Model: Compare Datasets (Group) across Marker Types
  # ------------------------------------------------------------------
  # interaction Group*Marker_Type tests if dataset differences vary by marker
  m_q2 <- lmer(Marker_Value ~ Group * Marker_Type + (1 | Participant), data = df_all)
  
  # Extract p-value for the interaction effect
  anova_res <- anova(m_q2, type = "3") 
  
  p_interaction <- anova_res["Group:Marker_Type", "Pr(>F)"]
  # p_group <- anova_res["Group", "Pr(>F)"]
  
  # Run Post-hoc to see exactly where datasets differ
  posthoc <- emmeans(m_q2, pairwise ~ Group | Marker_Type)
  posthoc_df <- as.data.frame(posthoc$contrasts)
  
  q2_results <- rbind(q2_results, data.frame(
    Density = as.numeric(density),
    P_Value = p_interaction
  ))
  
  posthoc_results <- rbind(posthoc_results, data.frame(
    Density = as.numeric(density),
    Marker = posthoc_df$Marker_Type,
    Estimate = posthoc_df$estimate, # Wahre Differenz (Beta)
    P_Value = posthoc_df$p.value
  ))
  
  print(paste("Processed density", density, "- Group:Marker P:", round(p_interaction, 5)))



  # ------------------------------------------------------------------
  # Q2 Model: Full vs. Null (Testing Random Effect Significance)
  # ------------------------------------------------------------------
  
  # 1. Your Full Model
  m_q2 <- lmer(Marker_Value ~ Group * Marker_Type + (1 | Participant), data = df_all)
  
  # 2. Likelihood Ratio Test for Random Effects
  # ranova() automatically creates the null model without (1|Participant) and compares them
  re_test <- ranova(m_q2)
  p_random_effect <- re_test[2, "Pr(>Chisq)"] 
  
  # 3. Extract Fixed Effects as you did before
  anova_res <- anova(m_q2, type = "3") 
  p_interaction <- anova_res["Group:Marker_Type", "Pr(>F)"]
  
  # --- Optional: Print the result to console to monitor variance ---
  print(paste("Density", density, "| RE (Participant) p-value:", round(p_random_effect, 5)))
  
  # Add to your results storage (expand your dataframe to keep track of this)
  q2_results <- rbind(q2_results, data.frame(
    Density = as.numeric(density),
    P_Value = p_interaction,
    P_Random_Effect = p_random_effect  # Store this to see if it changes across densities
  ))
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
  
  geom_line(color = "black", size = 1.2) +
  geom_point(color = "black", size = 3.5) +
  
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
    subtitle = "Group:Marker Effect Significance (Group)",
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
# Q2 Plot: Comparison of Fixed Interaction vs. Random Effect Significance
# ======================================================================================
library(tidyr)

# Daten für den Plot vorbereiten (Long Format)
# Wir nehmen an, dass q2_results nun die Spalten 'P_Value' (Interaction) 
# und 'P_Random_Effect' enthält.
plot_data <- q2_results %>%
  pivot_longer(cols = c(P_Value, P_Random_Effect), 
               names_to = "Effect_Type", 
               values_to = "P_Val_Data") %>%
  mutate(Effect_Label = ifelse(Effect_Type == "P_Value", 
                               "Group:Marker (Fixed)", 
                               "Participant (Random)"))

actual_min_p <- 1e-10
y_floor_q2 <- 10^floor(log10(actual_min_p))

# Y-Achsen-Breaks definieren
shared_y_breaks_q2 <- 10^seq(0, floor(log10(y_floor_q2)), by = -10)

p_comparison_plot <- ggplot(plot_data, aes(x = Density, y = P_Val_Data, color = Effect_Label)) +
  # Signifikanzschwellenwert (0.05)
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red", alpha = 0.6) +
  
  # Linien und Punkte
  geom_line(size = 1.2) +
  geom_point(size = 3.5) +
  
  # Log-Skala
  scale_y_log10(
    breaks = shared_y_breaks_q2,
    labels = scales::label_log(),
    expand = expansion(mult = c(0.1, 0.1))
  ) +
  
  # Farben definieren
  scale_color_manual(values = c("Group:Marker (Fixed)" = "black", 
                                "Participant (Random)" = "royalblue")) +
  
  scale_x_continuous(breaks = seq(0.1, 1.0, by = 0.1)) +
  coord_cartesian(ylim = c(y_floor_q2, 1)) + 
  
  labs(
    title = "LRT Significance: Fixed vs. Random Effects",
    subtitle = "Comparing beRNN vs beRNN original dataset",
    x = "Network Density",
    y = "p-value (Log Scale)",
    color = "Effect Type"
  ) +
  
  theme_bw() + 
  theme(
    aspect.ratio = 0.75, 
    panel.border = element_rect(colour = "black", fill = NA, size = 1.2),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    axis.title = element_text(face = "bold"),
    plot.title = element_text(size = 12, face = "bold")
  ) +
  
  # Label für die 0.05 Linie
  annotate("text", x = 0.1, y = 0.2, label = "p = 0.05", 
           color = "red", fontface = "italic", hjust = 0)

print(p_comparison_plot)



# ======================================================================================
# save Q2 plots - Dataset comparison
# ======================================================================================
output_filename_q2 <- paste0("LMM1_Q2_IE_beRNN_highDimCorrects_brain.png")
output_path_q2 <- file.path(folder, output_filename_q2)

ggsave(
  filename = output_path_q2,
  plot = p_q2_plot,
  width = 6,
  height = 3.5,
  dpi = 300,
  bg = "white"
)


# ======================================================================================
# save Q2 plots - Dataset comparison
# ======================================================================================
output_filename_q3 <- paste0("LMM_Q2_RE_beRNN_highDim_deadNfiltered.png")
output_path_q3 <- file.path(folder, output_filename_q3)

ggsave(
  filename = output_path_q3,
  plot = p_comparison_plot,
  width = 6,
  height = 3.5,
  dpi = 300,
  bg = "white"
)





# ======================================================================================
# Post-hoc Heatmap Plot
# ======================================================================================

if (nrow(posthoc_results) > 0) {
  # Reihenfolge fixieren (Modularity oben)
  posthoc_results$Marker <- factor(posthoc_results$Marker, 
                                   levels = rev(c("clustering", "modularity", "participation")))
  
  p_heatmap <- ggplot(posthoc_results, aes(x = factor(Density), y = Marker, fill = Estimate)) +
    geom_tile(color = "white", size = 0.5) +
    # Divergierende Skala: Blau (beRNN kleiner), Rot (beRNN größer)
    scale_fill_gradient2(low = "#0571b0", mid = "white", high = "#ca0020", 
                         midpoint = 0, name = "mean diff.") +
    # P-Werte als Text (Fett gedruckt)
    geom_text(aes(label = sprintf("%.3f", P_Value)), color = "black", size = 3.5, fontface = "bold") +
    labs(title = "Strukturelle Ähnlichkeit: beRNN vs. Brain",
         subtitle = "Zahlen = p-Werte | Farben = Wahre Differenz (Estimate)",
         x = "Network Density", y = "Topological Marker") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
          panel.grid = element_blank(),
          plot.title = element_text(face = "bold"))
  
  print(p_heatmap)
} else {
  message("Keine Ergebnisse zum Plotten vorhanden!")
}

print(p_heatmap)



# ======================================================================================
# Save Heatmap
# ======================================================================================
output_filename_heatmap <- "LMM1_Q2_PostHoc_IE_beRNN_highDimCorrects_brain.png"
output_path_heatmap <- file.path(folder, output_filename_heatmap)

ggsave(
  filename = output_path_heatmap,
  plot = p_heatmap,
  width = 8,
  height = 4,
  dpi = 300,
  bg = "white"
)


