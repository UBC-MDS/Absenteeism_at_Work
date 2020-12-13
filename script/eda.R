# author: Yiki Su
# date: 2020-11-26

"Creates eda plots for the pre-processed training data from the absenteeism data.
Saves the tables and plots as a feather and png files.

Usage: eda.R --train=<train> --out_dir=<out_dir>
  
Options:
--train=<train>     Path (including filename) to training data (which needs to be saved as a feather file)
--out_dir=<out_dir> Path to directory where the tables and plots should be saved
" -> doc

library(feather)
library(arrow)
library(tidyverse)
library(docopt)
library(ggthemes)
theme_set(theme_minimal())
library(ggplot2)
library(dplyr)
library(ggcorrplot)

opt <- docopt(doc)

main <- function(train, out_dir) {
  
  # read in the train dataframe
  train_data <- read_feather(train)
  
  # generate the plot for the correlation matrix
  train_data_copy <- train_data
  train_data_copy['Social drinker'] <- lapply(train_data_copy['Social drinker'], as.numeric)
  train_data_copy['Social smoker'] <- lapply(train_data_copy['Social smoker'], as.numeric)
  train_data_copy['Disciplinary failure'] <- lapply(train_data_copy['Disciplinary failure'], as.numeric)
  train_data_copy$Education <- as.numeric(as.character(train_data_copy$Education))
  train_data_copy$Seasons <- as.numeric(as.character(train_data_copy$Seasons))
  train_data_copy$`Month of absence` <- as.numeric(as.character(train_data_copy$`Month of absence`))
  train_data_copy$`Reason for absence` <- as.numeric(as.character(train_data_copy$`Reason for absence`))
  train_data_copy$`Day of the week` <- as.numeric(as.character(train_data_copy$`Day of the week`))
  
  corr_matrix <- train_data_copy %>% 
    select(-ID) %>% 
    cor() %>% 
    ggcorrplot() +
    theme(text =  element_text(size = 20))
  
  # save the correlation matrix plot
  ggsave(paste0(out_dir, "/correlation_matrix.png"), 
         corr_matrix,
         width = 10, 
         height = 10)
  
  # generate the distribution plot of each feature
  dist_plot <- train_data_copy %>%
    select(-ID) %>% 
    pivot_longer(everything()) %>%
    ggplot(aes(x=value)) +
    geom_histogram(bins = 50, fill = "blue") +
    facet_wrap(~name, ncol = 4, scales = 'free') +
    theme(text =  element_text(size = 12))
  
  # save the distribution plot
  ggsave(paste0(out_dir, "/distribution_plot.png"),
         dist_plot,
         width = 10,
         height = 10)
  
  # generate the frequency plot for "Reason for absence"
  reason_list = list(
    '0' = 'Unknown',
    '1' = 'Certain infectious and parasitic diseases',
    '2' = 'Neoplasms',
    '3' = 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',
    '4' = 'Endocrine, nutritional and metabolic diseases',
    '5' = 'Mental and behavioural disorders',
    '6' = 'Diseases of the nervous system',
    '7' = 'Diseases of the eye and adnexa',
    '8' = 'Diseases of the ear and mastoid process',
    '9' = 'Diseases of the circulatory system',
    '10' = 'Diseases of the respiratory system',
    '11' = 'Diseases of the digestive system',
    '12' = 'Diseases of the skin and subcutaneous tissue',
    '13' = 'Diseases of the musculoskeletal system and connective tissue',
    '14' = 'Diseases of the genitourinary system',
    '15' = 'Pregnancy, childbirth and the puerperium',
    '16' = 'Certain conditions originating in the perinatal period',
    '17' = 'Congenital malformations, deformations and chromosomal abnormalities',
    '18' = 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
    '19' = 'Injury, poisoning and certain other consequences of external causes',
    '20' = 'External causes of morbidity and mortality',
    '21' = 'Factors influencing health status and contact with health services',
    '22' = 'Patient follow-up',
    '23' = 'Medical consultation',
    '24' = 'Blood donation',
    '25' = 'Laboratory examination',
    '26' = 'Unjustified absence',
    '27' = 'Physiotherapy',
    '28' = 'Dental consultation'
  )
  train_data_copy <- rename(train_data_copy, reason_for_absence = 'Reason for absence')
  count_df <- train_data_copy %>% 
    group_by(reason_for_absence) %>% 
    summarise(Frequency = n())
  count_df$reason_for_absence <- as.character(count_df$reason_for_absence)
  count_df$reason_for_absence <- unlist(reason_list[count_df$reason_for_absence], use.names=FALSE)
  mean_count <- mean(count_df$Frequency)
  frequency_plot <- ggplot(count_df, aes(y=reorder(reason_for_absence,-Frequency), x=Frequency, fill ="red")) +
    geom_col() +
    geom_vline(xintercept = mean_count, color = "blue") +
    ggtitle("Number of occurrences for each reason for absence")+
    xlab("Number of occurrences") +
    scale_x_continuous(breaks = seq(0, 120, 10), expand = expansion(mult = c(0, 0.05))) +
    theme(panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank(),
          axis.title.y=element_blank(),
          legend.position = "none",
          plot.title = element_text(size=22),
          text =  element_text(size = 18)) 
  
  
  # save the frequency plot
  ggsave(paste0(out_dir, "/frequency_plot.png"),
         frequency_plot,
         width = 20,
         height = 10)

}

main(opt[["--train"]], opt[["--out_dir"]])