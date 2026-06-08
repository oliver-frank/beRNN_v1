library(dplyr)
library(lme4)

# read data
data_path <- "C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\all_tasks_flat_counts.csv"
all_data <- read.csv(data_path, stringsAsFactors = FALSE)

# calculate total number of errors for each task respectively
all_data <- all_data %>%
  group_by(task, subject) %>%
  mutate(total_errors = sum(count)) %>%
  ungroup()

unique_tasks <- unique(all_data$task)

for (current_task in unique_tasks) {
  cat("\n========================================================\n")
  cat("ANALYSE FOR TASK:", current_task, "\n")
  cat("========================================================\n")
  
  # filter data for current task
  task_data <- all_data %>% filter(task == current_task)
  
  # robust fixed effect GLM
  fixed_model <- glm(
    count ~ error_category * subject + offset(log(total_errors)), 
    data = task_data, 
    family = poisson(link = "log")
  )
  
  print(drop1(fixed_model, test = "Chisq"))
}



library(ggplot2)
library(dplyr)
library(viridis)
library(scales)

# prepare data
plot_data <- all_data %>%
  filter(task == "DM_Anti") %>% 
  mutate(
    percentage = (count / total_errors) * 100
  )

# change participant namin
plot_data <- plot_data %>%
  mutate(
    subject = case_when(
      subject == "beRNN_03" ~ "HC1",
      subject == "beRNN_04" ~ "HC2",
      subject == "beRNN_01" ~ "MDD",
      subject == "beRNN_02" ~ "ASD",
      subject == "beRNN_05" ~ "SCZ",
      TRUE ~ subject
    ),
    subject = factor(subject, levels = c("HC1", "HC2", "MDD", "ASD", "SCZ"))
  )

# Get error categories
top_10_categories <- plot_data %>%
  group_by(error_category) %>%
  summarise(grand_total = sum(count)) %>%
  arrange(desc(grand_total)) %>%
  slice_head(n = 10) %>%
  pull(error_category)

heatmap_data_top10 <- plot_data %>%
  filter(error_category %in% top_10_categories)

# create heatmap
my_heatmap <- ggplot(heatmap_data_top10, aes(x = subject, y = error_category, fill = percentage)) +
  geom_tile(color = "white", linewidth = 0.5) + 
  scale_fill_viridis(
    option = "plasma", 
    name = "", 
    limits = c(0, 50),
    oob = scales::squish,
    # HIER WIRD DIE LEGENDE KOMPAKT GEMACHT:
    guide = guide_colorbar(
      barwidth = unit(0.3, "cm"),  # Macht den Farbbalken deutlich schmaler
      barheight = unit(2.5, "cm"), # Macht den Farbbalken kürzer
      title.theme = element_text(size = 9, face = "bold"), # Kleinere Legendenschrift
      label.theme = element_text(size = 7)
    )
  ) +
  coord_fixed(ratio = 1) + 
  labs(
    x = "",
    y = ""
  ) +
  # Eine kleinere Basis-Textgröße (9 statt 11) lässt die Zellen größer wirken
  theme_minimal(base_size = 9) + 
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(size = 9, vjust = 0.5), # Größe für Probanden (z.B. 11)
    axis.text.y = element_text(size = 9),  
    axis.title = element_text(size = 9, face = "bold"),
    legend.box.spacing = unit(0.1, "cm") # Rückt die Legende näher an den Plot
  )

# export
ggsave(
  filename = "C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\heatmap_anti_top10.png",
  plot = my_heatmap,
  width = 5.5,   
  height = 5.0,  
  dpi = 300      
)



library(ggplot2)
library(dplyr)
library(viridis)
library(scales)
library(egg)

# read data
data_path <- "C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\all_tasks_flat_counts.csv"
all_data <- read.csv(data_path, stringsAsFactors = FALSE)

all_data <- all_data %>%
  group_by(task, subject) %>%
  mutate(total_errors = sum(count)) %>%
  ungroup()
# change naming
task_order_vertical <- c(
  'DM',      'DM_Anti',
  'EF',      'EF_Anti',
  'RP',      'RP_Anti',
  'RP_Ctx1', 'RP_Ctx2',
  'WM',      'WM_Anti',
  'WM_Ctx1', 'WM_Ctx2'
)
# prepare data
plot_data_all <- all_data %>%
  filter(task %in% task_order_vertical) %>%
  mutate(
    percentage = (count / total_errors) * 100,
    task = factor(task, levels = task_order_vertical) 
  ) %>%
  mutate(
    subject = case_when(
      subject == "beRNN_03" ~ "HC1",
      subject == "beRNN_04" ~ "HC2",
      subject == "beRNN_01" ~ "MDD",
      subject == "beRNN_02" ~ "ASD",
      subject == "beRNN_05" ~ "SCZ",
      TRUE ~ subject
    ),
    subject = factor(subject, levels = c("HC1", "HC2", "MDD", "ASD", "SCZ"))
  )

# sort top categories
heatmap_data_all_top5 <- plot_data_all %>%
  group_by(task, error_category) %>%
  summarise(grand_total = sum(count), .groups = "drop") %>%
  group_by(task) %>%
  arrange(desc(grand_total)) %>%
  slice_head(n = 5) %>%
  ungroup() %>%
  semi_join(plot_data_all, ., by = c("task", "error_category"))

big_heatmap <- ggplot(heatmap_data_all_top5, aes(x = subject, y = error_category, fill = percentage)) +
  geom_tile(color = "white", linewidth = 0.3) + 

  geom_text(
    aes(
      label = sprintf("%.0f", percentage),
      color = ifelse(percentage > 35 | percentage < 10, "white", "black")
    ),
    size = 1.6,
    fontface = "plain"
  ) +

  scale_color_identity() +

  scale_fill_gradient2(
    low = "#3b4cc0",      
    mid = "#e2e2e2",      
    high = "#b40426",     
    midpoint = 25,        
    name = "Share (%)", 
    limits = c(0, 50),
    oob = scales::squish,
    guide = guide_colorbar(
      barwidth = unit(0.3, "cm"),
      barheight = unit(3.5, "cm"),
      title.theme = element_text(size = 8, face = "bold"),
      label.theme = element_text(size = 7)
    )
  ) +
  
  facet_wrap(~task, scales = "free_y", ncol = 2) +
  labs(
    x = "Participant Group / Subject",
    y = NULL
  ) +
  theme_minimal(base_size = 9) + 
  theme(
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold", size = 10),
    strip.background = element_rect(fill = "#f0f0f0", color = NA),
    
    axis.text.x = element_text(size = 8, angle = 45, hjust = 1, vjust = 1), 
    axis.text.y = element_blank(), 
    
    axis.title.x = element_text(size = 10, face = "bold", margin = margin(t = 10)),
    legend.box.spacing = unit(0.3, "cm"),
    panel.spacing = unit(0.4, "cm"),
    plot.margin = margin(t = 0.2, r = 0.5, b = 0.8, l = 0.2, unit = "cm")
  )


