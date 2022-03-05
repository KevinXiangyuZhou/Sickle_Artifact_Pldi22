

library(dplyr)
library(ggplot2)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
# path to the result: '../output/paper_running_rlt_pldi.json'
# data <- fromJSON('../output/paper_running_rlt_pldi.json', flatten = TRUE)
data <- fromJSON(args[1], flatten = TRUE)


data <- as.data.frame(data)

dat <- data.frame(x = rnorm(100), y = rnorm(100))
number_ticks <- function(n) {function(limits) pretty(limits, n)}

process_data <- function(data) {
  new_data <- data
  colnames(new_data)<-gsub("data.","",colnames(new_data))
  colnames(new_data)<-sub("analysis_type","type",colnames(new_data))
  # filter out three baseline and count the program solved over time
  # data$with_pruning_or_not[data$with_pruning_or_not == 0] <- "enumeration"
  # data$with_pruning_or_not[data$with_pruning_or_not == 1] <- "with analysis"
  new_data$type[new_data$with_pruning_or_not == 0] <- "Type Abstraction"
  new_data$type[new_data$with_pruning_or_not == 1 & new_data$type == "trace"] <- "Provenance Abstraction (Ours)"
  new_data$type[new_data$with_pruning_or_not == 1 & new_data$type == "value"] <- "Value Abstraction"
  return(new_data)
}
# process data: combine with pruning or not with pruning type as type
data <- process_data(data)

# data <- data %>% mutate(md = median(input_cells))

data_filtered_e <- data %>% filter(data$num_program <= 2)
data_filtered_h <- data %>% filter(data$num_program > 2)

print(nrow(data_filtered_e) / 3)
print(nrow(data_filtered_h) / 3)
# 37 hard cases, 33 solvable
# 43 easy cases

data_with_a1 <- data_filtered_e %>% 
  filter(type =="Provenance Abstraction (Ours)") %>% 
  arrange(time) %>% 
  mutate(solved = row_number())
data_with_a2 <- data_filtered_h %>% 
  filter(type =="Provenance Abstraction (Ours)") %>% 
  arrange(time) %>% 
  mutate(solved = row_number())
trace_h_solved <- data_filtered_h %>% 
  filter(type =="Provenance Abstraction (Ours)", timeout == 0) %>% 
  nrow()
data_with_a2$solved[data_with_a2$timeout == 1] <- trace_h_solved

# data_with_a <- data_with_a %>%  mutate(solved = round(solved / nrow(data_with_a), 2) * 100)

# process time exp data
data_with_va1 <- data_filtered_e %>% 
  filter(type =="Value Abstraction") %>%
  arrange(time) %>% 
  mutate(solved = row_number())
data_with_va2 <- data_filtered_h %>% 
  filter(type =="Value Abstraction") %>%
  arrange(time) %>% 
  mutate(solved = row_number())
value_h_solved <- data_filtered_h %>% 
  filter(type =="Value Abstraction", timeout == 0) %>% 
  nrow()
data_with_va2$solved[data_with_va2$timeout == 1] <- value_h_solved

data_without_v1 <- data_filtered_e %>% 
  filter(type =="Type Abstraction") %>% 
  arrange(time) %>% 
  mutate(solved = row_number())
data_without_v2 <- data_filtered_h %>% 
  filter(type =="Type Abstraction") %>% 
  arrange(time) %>% 
  mutate(solved = row_number())
type_h_solved <- data_filtered_h %>% 
  filter(type =="Type Abstraction", timeout == 0) %>% 
  nrow()
data_without_v2$solved[data_without_v2$timeout == 1] <- type_h_solved

processed_data1 <- rbind(data_with_a1, data_with_va1, data_without_v1)
processed_data2 <- rbind(data_with_a2, data_with_va2, data_without_v2)
# ----------------------time exp----------------------


processed_data1_plot1 <- processed_data1
processed_data1_plot1 <- processed_data1_plot1 %>% add_row(type = "Provenance Abstraction (Ours)", with_pruning_or_not=TRUE, time = 0.01, solved=0)
processed_data1_plot1 <- processed_data1_plot1 %>% add_row(type = "Value Abstraction", with_pruning_or_not=TRUE, time = 0.01, solved=0)
processed_data1_plot1 <- processed_data1_plot1 %>% add_row(type = "Type Abstraction", with_pruning_or_not=FALSE, time = 0.01, solved=0)

processed_data1_plot1

plot_time_1 <- processed_data1_plot1 %>% ggplot() +
  geom_step(aes(color = type, 
                linetype = type,
                x = time, y = solved, 
                group = interaction(type, with_pruning_or_not)))  +
  # scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
  scale_color_manual(name = "Technique",
                     labels = c("Provenance Abstraction (Ours)", "Type Abstraction", "Value Abstraction"),
                     values = c("#F8766D", "#00BA38", "#619CFF")) +   
  scale_linetype_manual(name = "Technique",
                        labels = c("Provenance Abstraction (Ours)", "Type Abstraction", "Value Abstraction"),
                        values = c(1,5,6)) +
  labs(color = "Abstraction Type", x = "Time (seconds)", y = "# of solved benchmarks (easy)") +
  geom_hline(yintercept=43, linetype=3, 
             color = "dark gray") +
  scale_y_continuous(breaks = c(0, 5, 10, 15, 20, 25, 30, 35, 40, 43), labels = c(0, 5, 10, 15, 20, 25, 30, 35, 40, 43)) +
  scale_x_continuous(trans='log10',breaks = c(0.5, 1, 2, 4, 8, 24, 72, 128), labels = c("0.5", "1", "2", "4", "8", "24", "72", "128")) +
  theme(text = element_text(size=18),
        legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        panel.background = element_blank(),
        panel.border = element_rect(linetype = "solid", color="gray10", fill = NA),
        legend.position = c(0.72, 0.14),
        legend.background = element_rect(fill="white",
                                         size=0.09, linetype="solid", 
                                         colour ="gray"),
        axis.line = element_line(colour = "black"),
        strip.background=element_rect(fill="white"),
        strip.placement = "outside")

dat <- data.frame(x = rnorm(100), y = rnorm(100))
number_ticks <- function(n) {function(limits) pretty(limits, n)}


processed_data1_plot2 <- processed_data2
processed_data1_plot2 <- processed_data1_plot2 %>% add_row(type = "Provenance Abstraction (Ours)", with_pruning_or_not=TRUE, time = 0.01, solved=0)
processed_data1_plot2 <- processed_data1_plot2 %>% add_row(type = "Value Abstraction", with_pruning_or_not=TRUE, time = 0.01, solved=0)
processed_data1_plot2 <- processed_data1_plot2 %>% add_row(type = "Type Abstraction", with_pruning_or_not=FALSE, time = 0.01, solved=0)

plot_time_2 <- processed_data1_plot2 %>% ggplot() +
  geom_step(aes(color = type,
                linetype = type,
                x = time, y = solved, 
                group = interaction(type, with_pruning_or_not)))  +
  # scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
  scale_color_manual(name = "Technique",
                     labels = c("Provenance Abstraction (Ours)", "Type Abstraction", "Value Abstraction"),
                     values = c("#F8766D", "#00BA38", "#619CFF")) +   
  scale_linetype_manual(name = "Technique",
                        labels = c("Provenance Abstraction (Ours)", "Type Abstraction", "Value Abstraction"),
                        values = c(1,5,6)) +
  labs(color = "Abstraction Type", x = "Time (seconds)", y = "# of solved benchmarks (hard)") +
  geom_hline(yintercept=37, linetype=3, 
             color = "dark gray") +
  geom_hline(yintercept=trace_h_solved, linetype=3, 
             color = "light gray") +
  scale_y_continuous(breaks = c(0, 5, 10, 15, 20, 25, 30, 37), labels = c(0, 5, 10, 15, 20, 25, 30, 37)) +
  scale_x_continuous(trans='log10',breaks = c(1, 10, 60, 120, 240, 600), labels = c(1, "10", "60", "120", "240", "600")) +
  theme(text = element_text(size=18),
        legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        panel.background = element_blank(),
        panel.border = element_rect(linetype = "solid", color="gray10", fill = NA),
        legend.position = c(0.27, 0.74),
        legend.background = element_rect(fill="white",
                                         size=0.09, linetype="solid", 
                                         colour ="gray"),
        axis.line = element_line(colour = "black"),
        strip.background=element_rect(fill="white"),
        strip.placement = "outside")

# ----------------------plot num searched exp----------------------
# data_level_3 <- data %>% filter(data$num_program > 3)
# data_level_12 <- data %>% filter(data$num_program <= 3 & data$time < 10 )
plot_num_searched <- data_filtered_h %>% ggplot(aes(x=type,
                                                    y=num_program_visited, 
                                                    fill=type)) + 
  geom_boxplot(width=0.5, alpha = 0.7) + 
  geom_jitter(width=0.3, size = 1.3) +
  # scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
  labs(color = "Abstraction Type", x = "", y = "# of programs explored") +
  scale_fill_manual(name = "Technique",
                    labels = c("Provenance Abstraction", "Type Abstraction", "Value Abstraction"),
                    values = c("#F8766D", "#00BA38", "#619CFF")) +
  scale_x_discrete(labels = c("Provenance Abstraction", "Type Abstraction", "Value Abstraction")) +
  scale_y_continuous(breaks=number_ticks(6)) +
  theme(text = element_text(size=18),
        legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        panel.background = element_blank(),
        panel.border = element_rect(linetype = "solid", color="gray10", fill = NA),
        legend.position = c(0.78, 0.88),
        legend.background = element_rect(fill="white",
                                         size=0.05, linetype="solid", 
                                         colour ="gray"),
        axis.line = element_line(colour = "black"),
        strip.background=element_rect(fill="white"),
        strip.placement = "outside")

plot_num_searched_1 <- data_filtered_e %>% ggplot(aes(x=type,
                                                      y=num_program_visited, 
                                                      fill=type)) + 
  geom_boxplot(width=0.5, alpha = 0.7) + 
  geom_jitter(width=0.3, size = 1.3) +
  labs(color = "Abstraction Type", x = "", y = "# of programs explored") +
  coord_cartesian(ylim = c(0, 2000)) +
  scale_fill_manual(name = "Technique",
                    labels = c("Provenance Abstraction", "Type Abstraction", "Value Abstraction"),
                    values = c("#F8766D", "#00BA38", "#619CFF")) +
  scale_x_discrete(labels = c("Provenance Abstraction", "Type Abstraction", "Value Abstraction")) +
  scale_y_continuous(breaks=number_ticks(6)) +
  theme(text = element_text(size=18),
        legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        panel.background = element_blank(),
        panel.border = element_rect(linetype = "solid", color="gray10", fill = NA),
        legend.position = c(0.77, 0.87),
        legend.background = element_rect(fill="white",
                                         size=0.05, linetype="solid", 
                                         colour ="gray"),
        axis.line = element_line(colour = "black"),
        strip.background=element_rect(fill="white"),
        strip.placement = "outside")
# print(plot_size_exp)
png("./artifact/output/figure.13(1).png")
print(plot_num_searched)
dev.off()
png("./artifact/output/figure.13(2).png")
print(plot_num_searched_1)
dev.off()
png("./artifact/output/figure.12(1).png")
print(plot_time_1)
dev.off()
png("./artifact/output/figure.12(2).png")
print(plot_time_2)
dev.off()
