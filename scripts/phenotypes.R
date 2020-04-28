### summarize the phenotypes
library(tidyverse)
### move images to test and train sets
## ll svs_files > svs_files.txt 

svs_files <- read_delim("data/svs_files.txt", delim = "\t", col_names = F) %>% 
  separate(., X1, into = c("svs_id", "tail"), sep = "-01Z-00-DX", remove = F) %>% 
  distinct(.,svs_id, X1)

status <- read_csv("data/clinical.csv") %>% 
  select(submitter_id, gender, vital_status ) %>% 
  filter(vital_status != "Not Reported") %>% 
  distinct() %>% 
  left_join(.,svs_files, by = c("submitter_id" = "svs_id"))



## 75% of the sample size
smp_size <- floor(0.75 * nrow(status))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(status)), size = smp_size)

train_dead <- status[train_ind, ] %>% filter(vital_status == "Dead") %>% select(X1)
train_alive <- status[train_ind, ] %>% filter(vital_status == "Alive") %>% select(X1)
test_dead <- status[-train_ind, ] %>% filter(vital_status == "Dead") %>% select(X1)
test_alive <- status[-train_ind, ] %>% filter(vital_status == "Alive") %>% select(X1)


write_csv(train_dead, path = "data/train_dead.txt", col_names = F)
write_csv(train_alive, path = "data/train_alive.txt", col_names = F)
write_csv(test_dead, path = "data/test_dead.txt", col_names = F)
write_csv(test_alive, path = "data/test_alive.txt", col_names = F)


# while read file; do mv "svs_file/$file" test/alive/; done < test.alive.txt

# convert images
python3.6 wsi_svs_to_jpeg_tiles.py  -i /test/dead -o /test/dead
python3.6 wsi_svs_to_jpeg_tiles.py  -i /svs_files2 -o /svs_files2

