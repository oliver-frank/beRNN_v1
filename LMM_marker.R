# ======================================================================================
# Linear Mixed Model - FIXED VERSION
# ======================================================================================
if (!require("pacman")) install.packages("pacman")
pacman::p_load(jsonlite, dplyr, tidyr, lme4, lmerTest, emmeans, ggplot2, performance, patchwork)

topMarker_list <- c('clustering', 'modularity', 'participation')
densities <- c('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0')
folder <- "C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels/__topologicalMarker_pValue_lists"
modelsets <- c('topologicalMarker_dict_beRNN_highDim_', 
               'topologicalMarker_dict_beRNN_highDim_correctOnly_',
               'topologicalMarker_dict_brain_')

q1_results <- data.frame()
q2_results <- data.frame()
posthoc_results_q1 <- data.frame()

# Hilfsfunktion
parse_json_to_df <- function(file_path, group_label) {
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
            Pair_ID = as.factor(m_idx), 
            Marker_Type = marker_raw,
            Marker_Value = values[m_idx]
          )
        }
      }
    }
  }
  return(bind_rows(rows))
}

for (density in densities) {
  path_1 <- file.path(folder, paste0(modelsets[2], density, ".json"))
  path_3 <- file.path(folder, paste0(modelsets[2], density, ".json"))
  
  if (!file.exists(path_1) || !file.exists(path_3)) next
  
  df_all <- bind_rows(parse_json_to_df(path_1, "beRNN"), 
                      parse_json_to_df(path_3, "brain")) %>%
    mutate(across(c(Participant, Group, Marker_Type), as.factor))
  
  # --- Q1 Model ---
  m_q1 <- lmer(Marker_Value ~ Group * Marker_Type + (1 | Participant), data = df_all)
  
  # Check for singularity (var=0)
  is_singular <- isSingular(m_q1)
  
  # ANOVA 
  anova_df <- as.data.frame(anova(m_q1, type = "3"))
  p_col <- grep("Pr", names(anova_df), value = TRUE)
  p_interaction <- if(length(p_col) > 0) anova_df["Group:Marker_Type", p_col] else NA
  
  # ICC 
  curr_icc <- 0
  if (!is_singular) {
    icc_obj <- performance::icc(m_q1)
    curr_icc <- as.numeric(icc_obj[[1]]) 
  } else {
    message(paste("Density", density, ": Model is singular (ICC set to 0)"))
  }
  
  q1_results <- rbind(q1_results, data.frame(
    Density = as.numeric(density), 
    P_Value = p_interaction, 
    ICC = curr_icc,
    Singular = is_singular
  ))
  
  # Post-hoc
  ph <- as.data.frame(emmeans(m_q1, pairwise ~ Group | Marker_Type)$contrasts)
  ph$Density <- as.numeric(density)
  posthoc_results_q1 <- rbind(posthoc_results_q1, ph)
  
  # --- Q2 Model ---
  m_q2 <- lmer(Marker_Value ~ Marker_Type + (1 | Participant), data = df_all)
  p_re <- ranova(m_q2)[2, "Pr(>Chisq)"]
  
  q2_results <- rbind(q2_results, data.frame(
    Density = as.numeric(density),
    P_Random_Effect = p_re
  ))
  
  print(paste("Density", density, "done."))
}



# ======================================================================================
# Q1 Plot: Dataset Comparison (beRNN vs Brain)
# ======================================================================================

actual_min_p <- 1e-100
y_floor_q1 <- 10^floor(log10(actual_min_p)) 

# define y-ticks
shared_y_breaks_q1 <- 10^seq(1e-20, floor(log10(y_floor_q1)), by = -10)
shared_y_breaks_q1 <- sort(unique(c(shared_y_breaks_q1)))

p_q1_plot <- ggplot(q1_results, aes(x = Density, y = P_Value)) +
  # red significance line
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red", alpha = 0.6) +
  
  geom_line(color = "black", size = 1.2) +
  geom_point(color = "black", size = 3.5) +
  
  # log-scale
  scale_y_log10(
    breaks = shared_y_breaks_q1,
    labels = scales::label_log(),
    expand = expansion(mult = c(0.1, 0.1)) 
  ) +
  
  scale_x_continuous(breaks = seq(0.1, 1.0, by = 0.1)) +
  
  # define visual space
  coord_cartesian(ylim = c(y_floor_q1, 1)) + 
  
  labs(
    title = "Comparison: ANN vs Brain",
    subtitle = "Group:Marker Effect Significance",
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

print(p_q1_plot)



# ======================================================================================
# Q1 Combined Plot: Mean Differences & ICC (Similarity)
# ======================================================================================
library(patchwork)

shared_theme <- theme_bw() + 
  theme(
    aspect.ratio = 0.25,          # y-axis
    panel.grid.minor = element_blank(),
    axis.title = element_text(face = "bold", size = 9),
    plot.title = element_text(face = "bold", size = 10, margin = margin(b = 0)),
    legend.position = "right",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7),
    plot.margin = margin(t = 2, r = 5, b = 2, l = 5) 
  )

# Mean Difference
p_diff <- ggplot(posthoc_results_q1, aes(x = Density, y = estimate, color = Marker_Type)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.4) +
  geom_line(size = 1.5) +            
  geom_point(size = 2.5) + 
  scale_x_continuous(breaks = seq(0.1, 1, 0.1), labels = NULL) + 
  labs(title = "Mean Differences: beRNN vs. Brain", y = "Difference", x = NULL) +
  shared_theme

# ICC (equivalence)
p_icc <- ggplot(q1_results, aes(x = Density, y = ICC)) + 
  geom_area(fill = "steelblue", alpha = 0.3) +
  geom_line(color = "steelblue", size = 1.5) + 
  geom_point(color = "steelblue", size = 2.5) +
  scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
  scale_x_continuous(breaks = seq(0.1, 1, 0.1)) +
  labs(title = "Similarity: Within-Participant ICC", y = "ICC (0-1)", x = "Density") +
  shared_theme

p_final <- (p_diff / p_icc) + 
  plot_layout(guides = "collect") & 
  theme(
    plot.margin = margin(0, 0, 0, 0),
    panel.spacing = unit(0.2, "lines") 
  )

print(p_final)



# ======================================================================================
# Q2 Plot: Comparison within group between participant with Random Effect Significance
# ======================================================================================
library(tidyr)

plot_data <- q2_results %>%
  pivot_longer(cols = P_Random_Effect, 
               names_to = "Effect_Type", 
               values_to = "P_Val_Data")

actual_min_p <- 1e-15
y_floor_q2 <- 10^floor(log10(actual_min_p))

shared_y_breaks_q2 <- 10^seq(0, floor(log10(y_floor_q2)), by = -15)

p_q2_plot <- ggplot(plot_data, aes(x = Density, y = P_Val_Data)) +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red", alpha = 0.6) +
  
  geom_line(size = 1.2, color = "royalblue") +
  geom_point(size = 3.5, color = "royalblue") +

  scale_y_log10(
    breaks = shared_y_breaks_q2,
    labels = scales::label_log(),
    expand = expansion(mult = c(0.1, 0.1))
  ) +
  
  scale_x_continuous(breaks = seq(0.1, 1.0, by = 0.1)) +
  coord_cartesian(ylim = c(y_floor_q2, 1)) + 
  
  labs(
    title = "LRT Significance: Random Effects",
    subtitle = "Comparing ANN vs ANN",
    x = "Network Density",
    y = "p-value (Log Scale)"
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

print(p_q2_plot)



# ======================================================================================
# save Q1 plot - Dataset comparison
# ======================================================================================
output_filename_q1 <- paste0("LMM_Q1_IE_beRNN_highDimCorrects_brain.png")
output_path_q1 <- file.path(folder, output_filename_q1)

ggsave(
  filename = output_path_q1,
  plot = p_q1_plot,
  width = 6,
  height = 3.5,
  dpi = 300,
  bg = "white"
)



# ======================================================================================
# save Q1 plot1 - Mean Differences & ICC (Similarity)
# ======================================================================================
output_filename_q1 <- paste0("LMM_Q1_DIFF_ICC_beRNN_highDimCorrects_brain.png")
output_path_q1 <- file.path(folder, output_filename_q1)

ggsave(
  filename = output_path_q1,
  plot = p_final,
  width = 6,
  height = 3.5,
  dpi = 300,
  bg = "white"
)



# ======================================================================================
# save Q2 plots - Random Effect
# ======================================================================================
output_filename_q2 <- paste0("LMM_Q2_RE_beRNN_highDimCorrects.png")
output_path_q2 <- file.path(folder, output_filename_q2)

ggsave(
  filename = output_path_q2,
  plot = p_q2_plot,
  width = 6,
  height = 3.5,
  dpi = 300,
  bg = "white"
)


