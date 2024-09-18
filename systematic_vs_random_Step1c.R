# systematic vs random 

path <- "/Users/marcschubert/Documents/rnns/Data/BeRNN_01/PreprocessedData_wResp_ALL_V3/all_error_counts.csv"
setwd("/Users/marcschubert/Documents/rnns/Data/BeRNN_01/PreprocessedData_wResp_ALL_V3/ErrorCats")
data <- read.csv(path)
data$fineGrained <- str_detect(data$taskgroup, "fineGrained")
data$fineGrained[data$fineGrained] <- "fineGrained"
data$fineGrained[data$fineGrained=="FALSE"] <- "basic"
data$error_name <- paste0(data$taskgroup, "_", data$error_name)



ggplot(data %>% 
         filter(count > 0), aes(count)) + 
  geom_histogram() + 
  facet_wrap(~fineGrained) + 
  ggtitle("Error counts - (0s removed)")
ggsave("error_counts.png")


ggplot(data %>% 
         filter(fineGrained == "fineGrained") %>%
         filter(count > 0), aes(count, fill = taskgroup)) + 
  geom_histogram() + 
  facet_wrap(~taskgroup, ncol = 2) + 
  ggtitle("Error counts- fine grained (0s removed)")
ggsave("error_counts split by taske finegrained.png")



ggplot(data %>% 
         filter(fineGrained == "basic") %>%
         filter(count > 0), aes(count, fill = taskgroup)) + 
  geom_histogram() + 
  facet_wrap(~taskgroup, ncol = 2) + 
  ggtitle("Error counts- basic (0s removed)")
ggsave("error_counts split by task.png")


ggplot(data %>% 
         filter(count > 0)
         , aes(y = count, x = reorder(error_name, count))) + 
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_blank()) +
  facet_wrap(~fineGrained, nrow = 2, 
             scales="free") +
ggtitle("Occurence of errors ordered")
ggsave("errorcounts_split.png")



data <- data %>% 
  group_by(taskgroup, fineGrained) %>% 
  mutate(n_pergroup = n(), 
         sum_pergroup = sum(count), 
         expected = sum_pergroup / n_pergroup, 
         perc = count / sum_pergroup,
         std_res = (count-expected) / sqrt(expected+1)
         )


basic <- data
basic <- basic %>%
  filter(fineGrained == "basic")


ggplot(basic, aes(x =count,  y = perc)) + 
  geom_point()
basic$error_name
p1 <- ggplot(basic %>%
         filter(count>0)
         , aes(y =count,x = reorder(error_name, count), fill = perc)) + 
  theme_bw() + 
  theme(axis.text.x = element_blank())  + 
  scale_fill_viridis_c()+
  geom_bar(stat = "identity")  + 
  ggtitle("Error Count, Percentage per task group, basic")
p1
ggsave("error_count, percentage_per_taskgroup.png")  


p2 <- ggplot(data %>% 
         filter(count>0)
       , aes(y =count,x = reorder(error_name, count), fill = perc>0.1)) + 
  theme_bw() + 
    theme(axis.text.x = element_blank())  + 
  geom_bar(stat = "identity")  + 
  ggtitle("Error Count, Percentage per task group")
p2
p3 <- ggplot(basic %>% 
         filter(count>0)
       , aes(y =count,x = reorder(error_name, count), fill = perc>0.05)) + 
  theme_bw() + 
  theme(axis.text.x = element_blank())  + 
  geom_bar(stat = "identity")  + 
  ggtitle("Error Count, Percentage per task group")


ggarrange(p1,p2,p3, ncol = 1)
ggsave("percentage_per_taskgroup.png")



#############
data$fineGrained
xx <- data %>%
  filter(fineGrained == "basic")
xx$perc[is.na(xx$perc)] <- 0
xx <- xx %>% 
  filter(count >0)
table(xx$perc > 0.1)
xx


ggplot(xx, aes(x =taskgroup, y = count, fill = perc>0.1 )) + 
  geom_bar(stat = "identity", color="black") + 
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 45, hjust=1,vjust=1))
ggsave("fehlerclassen_nach_groups.png")

xx %>% 
  group_by(perc>0.1) %>% 
  summarize(n = sum(count))

ggplot(xx %>% 
  filter(taskgroup == "errors_dict_RP_Ctx2.json") , aes(x="",y = count, 
                                                        
                                                        fill = error_name)) + 
  geom_bar(stat = "identity", color = "black")  + NoLegend()
ggsave("fehlerclassen_nach_groups_RPctx2.png")


ggplot(xx, aes(x =taskgroup, y = count, fill = error_name )) + 
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle = 45, hjust=1,vjust=1), 
        legend.position = "none")
ggsave("fehlerclassen_nach_groups2.png")

ggplot(xx, aes(perc)) + 
  geom_histogram(color = "black", fill="grey") + 
  theme_bw()  + 
  geom_vline(xintercept = 0.1, color="darkblue")
ggsave("systematic_vs_random.png")

  




data$systematic_error <- data$perc > 0.05
systematic_errors <- basic$error_name[basic$systematic_error]


systematic_errors[!is.na(systematic_errors)]

write.csv(data, "ERROR_FILTERING.csv")

table(basic$perc >=0.05)

nrow(basic)


View(basic)








