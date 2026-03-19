# ======================================================================================
# Linear Mixed Model to compare behavioral data
# ======================================================================================
library(jsonlite)
library(tidyverse)
library(lme4)
library(lmerTest) # Erweitert lme4 um p-Werte
library(emmeans)  # Für Post-hoc-Tests

# Funktion zum Transformieren der JSON-Struktur
process_json_robust <- function(filepath, value_name) {
  # JSON einlesen
  raw_data <- fromJSON(filepath, simplifyVector = FALSE)
  
  # Schrittweise in Dataframe umwandeln
  df <- enframe(raw_data, name = "Participant", value = "Months") %>%
    unnest_longer(Months, indices_to = "Month") %>%
    unnest_longer(Months, indices_to = "Task") %>%
    # Sicherstellen, dass jeder Task-Inhalt eine flache Liste von Werten ist
    mutate(Values = map(Months, as.numeric)) %>% 
    select(-Months) %>%
    # Jetzt sicher entpacken
    unnest_longer(Values, indices_to = "Trial") %>%
    rename(!!value_name := Values)
  
  return(df)
}

# Daten laden und zusammenführen 
df_rt <- process_json_robust("W:/group_csp/analyses/oliver.frank/Data/participants_dict_reactionTime.json", "ReactionTime")
df_acc <- process_json_robust("W:/group_csp/analyses/oliver.frank/Data/participants_dict_accuracy.json", "Accuracy")
df_comp <- process_json_robust("W:/group_csp/analyses/oliver.frank/Data/participants_dict_complexities.json", "Complexity")

df_final <- df_rt %>%
  left_join(df_acc, by = c("Participant", "Month", "Task", "Trial")) %>%
  left_join(df_comp, by = c("Participant", "Month", "Task", "Trial"))

# Monat als Zahl/Faktor formatieren
df_final <- df_final %>%
  mutate(Month = as.numeric(Month))

df_final <- df_final %>%
  group_by(Participant, Month, Task) %>% 
  fill(Complexity, .direction = "down") %>% # Füllt die NAs mit dem Wert aus Zeile 1
  ungroup() %>%
  mutate(
    Month = as.factor(Month),
    Complexity = as.factor(Complexity)
  )



# Modelle erstellen
# Participant ist hier ein Random Effect (1|Participant)
model_rt <- lmer(ReactionTime ~ Month + Complexity + (1|Participant), data = df_final)
model_acc <- lmer(Accuracy ~ Month + Complexity + (1|Participant), data = df_final)

# Ergebnisse prüfen
summary(model_rt)
anova(model_rt)

# Post-hoc-Tests (Tukey)
# Vergleich der Monate
posthoc_rt <- emmeans(model_rt, pairwise ~ Month, pbkrtest.limit = 5140)
posthoc_acc <- emmeans(model_acc, pairwise ~ Month, pbkrtest.limit = 5140)

# Ergebnisse anzeigen
print(posthoc_rt$contrasts)
print(posthoc_acc$contrasts)

# Konstanz prüfen (Deskriptiv)
# Wo ist die Varianz am geringsten?
stability_check <- df %>%
  group_by(Month, Complexity) %>%
  summarise(SD_RT = sd(ReactionTime), .groups = 'drop')



