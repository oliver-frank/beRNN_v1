library(strex)

source("source.r")
setwd("/Users/marcschubert/Documents/rnns/models/XModel_150_BeRNN_01_Month_2-6_0.1Threshold")

files = list.files(pattern = ".csv")

data <- lapply(files, read.csv)
names(data) <- files

for (i in 1:length(data)) {
  name <- names(data)[i]
  data[[i]]$table <- name
  
}
bound <- do.call("rbind", data)

head(bound)
bound$training_set <- str_before_last(bound$table, "_")
bound$metric <- str_remove(str_after_last(bound$table, "_"), ".csv")
table(bound$metric)
bound$task <- str_after_first(bound$group, "_")
#bound$training_set <- factor(bound$training_set, levels = c("ORIGINAL", "ERRORS_REMOVED", "ERRORS_ONLY"))
bound$training_set <- str_remove(bound$training_set, "ORIGINAL_")
bound$training_set_names <- bound$training_set
bound$training_set_names[bound$training_set == "org"] <- "Correct+Systematic+Random (Orig)"
bound$training_set_names[bound$training_set == "corsys"] <- "Correct+Systematic"
bound$training_set_names[bound$training_set == "corrand"] <- "Correct+Random"
bound$training_set_names[bound$training_set == "coronly"] <- "Correct"
bound$training_set_names[bound$training_set == "sysonly"] <- "Systematic"
bound$training_set_names[bound$training_set == "randonly"] <- "Random"
bound$training_set_names[bound$training_set == "sysrand"] <- "Systematic+Random"
table(bound$training_set_names)

bound$training_set_names <- factor(bound$training_set_names, levels = c(
  "Correct+Systematic+Random (Orig)",
  "Correct+Systematic",
  "Correct+Random",
  "Correct",
  "Systematic",
  "Random",
  "Systematic+Random"
))

trainingset_cols <- c("black", 
            "blue1", 

            "yellow",
            "green",
            "purple",
                        "red",
            "orange"
            )



cur_metric <- "perf"
p1 <- ggplot(bound %>% 
         filter(metric == cur_metric)
         , aes(x = trials, y = values, color = training_set_names)) + 
  geom_smooth() + 
  scale_color_manual(values = trainingset_cols, name = "")+
  theme_bw() + 
  ggtitle(cur_metric)
cur_metric <- "cost"
p2 <- ggplot(bound %>% 
               filter(metric == cur_metric)
             , aes(x = trials, y = values, color = training_set_names)) + 
  geom_smooth() + 
  scale_color_manual(values = trainingset_cols, name = "")+
  
  theme_bw() + 
  ggtitle(cur_metric)


cur_metric <- "creg"

p3 <- ggplot(bound %>% 
               filter(metric == cur_metric)
             , aes(x = trials, y = values, color = training_set_names)) + 
  geom_smooth() + 
  scale_color_manual(values = trainingset_cols, name = "")+
  
  theme_bw() + 
  ggtitle(cur_metric)

ggarrange(p1, p2,p3, 
          ncol = 3,
          common.legend=T, legend="right")
ggsave("avg_metrics_per_trainingset_perf.png")


table(bound$training_set)
cur_metric <- "perf"
pa <- ggplot(bound %>% 
         filter(metric == cur_metric)%>%
         filter(training_set %in% c("coronly"  ,"corrand"  , "corsys"  ,    "org")  ), aes(x = trials, y = values, color = training_set_names)) + 
  geom_smooth() + 
  scale_color_manual(values = trainingset_cols[], name = "")+
  theme_bw() + 
  ggtitle(cur_metric)+ 
  ylim(c(0,0.5)) + 
  theme(legend.position = "bottom")

pb <- ggplot(bound %>% 
               filter(metric == cur_metric)%>%
               filter(!training_set %in% c("coronly"  ,"corrand"  , "corsys"  ,    "org")  ), aes(x = trials, y = values, color = training_set_names)) + 
  geom_smooth() + 
  scale_color_manual(values = trainingset_cols[-c(1:4)], name = "")+
  theme_bw() + 
  ggtitle(cur_metric) + 
  ylim(c(0,0.5))+ 
  theme(legend.position = "bottom")


ggarrange(pa,pb)
ggsave("PERF_corrects, errors.png")

bound


cur_metric <- "creg"
ggplot(bound %>% filter(metric == cur_metric), 
       aes(x = trials, y = values, color = task)) + 
  geom_smooth() + 
  theme_bw()+
  facet_wrap(~training_set + metric)

cur_metric <- "cost"
ggplot(bound %>% filter(metric == cur_metric), 
       aes(x = trials, y = values, color = task)) + 
  geom_smooth() + 
  theme_bw()+
  facet_wrap(~training_set + metric)

cur_metric <- "perf"
ggplot(bound %>% filter(metric == cur_metric, 
                        ), 
       aes(x = trials, y = values, color = task)) + 
  geom_line() + 
  theme_bw()+
  facet_wrap(~training_set_names + metric)
ggsave("performance_split_by_task.png")


cur_metric = "perf"
ggplot(bound %>% filter(metric == cur_metric,
                        !task %in% c("min", "avg") 
), 
aes(x = trials, y = values, color = training_set_names)) + 
  geom_line() + 
  scale_color_manual(values = trainingset_cols, name = "")+
  geom_smooth() + 
  theme_bw()+
  facet_wrap(~task)
ggsave("performance_each_task_separate.png")

bound$wm_notwm <- str_detect(bound$task, "WM")
bound$wm_notwm[bound$wm_notwm] <- "WM_Tasks"
bound$wm_notwm[bound$wm_notwm == "FALSE"] <- "Other_Tasks"
bound$wm_notwm[bound$task == "min"] <- "min"
bound$wm_notwm[bound$task == "avg"] <- "avg"
ggplot(bound %>% 
         filter(metric == "perf") %>%
         filter(wm_notwm %in% c("WM_Tasks", "Other_Tasks"))
         , aes(x = trials, values, color = wm_notwm)) + 
  #geom_line()+
  geom_smooth() + 
  facet_wrap(~training_set_names) + 
  theme_bw() + 
  ggtitle("WM Tasks vs Other Tasks")
ggsave("WM Tasks vs other tasks.png")



View(bound)
