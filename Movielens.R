###################################
# 1. Create edx set and validation set
###################################

# Note: this process could take a couple of minutes
library(kableExtra)
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1) # if using R 3.6.0: set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###################################
# 2.1 Exploratory data analysis
###################################


#Number of row and columns
nrow(edx)
ncol(edx)

#Number of different users and movies

edx %>% summarize(n_users = n_distinct(userId),n_movies = n_distinct(movieId))

#Movie Rating
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Movies Count") +
  ggtitle("Number of ratings per movie") + theme(plot.title = element_text(hjust = 0.5))


###################################
# 2.2.1 Basic model
###################################


#Define LOSS function RMSE

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2,na.rm=TRUE))
  
}


# Create the index and test/training set
edx <- edx %>% select(userId, movieId, rating)
test_index <- createDataPartition(edx$rating, times = 1, p = .2, list = F)
# Create the index
train <- edx[-test_index, ] 
test <- edx[test_index, ]
test <- test %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

test_dimension<-c(dim(train),dim(test))

# Mean accross all movies
mu_hat <- mean(train$rating) 
# RMSE of test set
RMSE_base <- RMSE(validation$rating, mu_hat) 
RMSE_base

rmse_table_val <- tibble(Method = "Base", RMSE = RMSE_base)
rmse_table_val %>% knitr::kable(caption = "RMSEs")


###################################
# 2.2.2 User and Movie effect Model
###################################


mu <- mean(train$rating)
movie_avgs <- train %>%
  group_by(movieId) %>%
  summarize(m_i = mean(rating - mu))
user_avgs <- test %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(u_i = mean(rating - mu - m_i))
predicted_ratings <- test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + m_i + u_i) %>% .$pred
model_RMSE <- RMSE(test$rating,predicted_ratings )
model_RMSE  


#User and Movie effect Model on validation set

validation <- validation %>% select(userId, movieId, rating)

mu <- mean(train$rating)

movie_avgs <- train %>%
  group_by(movieId) %>%
  summarize(m_i = mean(rating - mu))

user_avgs <- test %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(u_i = mean(rating - mu - m_i))

predicted_ratings <- test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + m_i + u_i) %>% .$pred

predicted_val <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + m_i + u_i) %>% .$pred

val_RMSE2 <- RMSE( validation$rating,predicted_val)
val_RMSE2
rmse_table_val <- bind_rows(rmse_table_val,
                            tibble(Method="User and Movie Effect ",  
                                   RMSE =val_RMSE2 ))


##############################################
# 2.2.3  Regularized user and Movie effect Model 
##############################################

lambdas <- seq(0, 10, 0.25)
# Sequence of lambdas to use
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(validation$rating,predicted_ratings))
})
## For each lambda, we find the b_i and the b_u, then make our prediction and test and plot.  
qplot(lambdas, rmses)  
lambda <- lambdas[which.min(rmses)]
lambda

movie_avgs_reg <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())
## Using lambda, find the movie effects
user_avgs_reg <- edx %>% 
  left_join(movie_avgs_reg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_u = n())
## Using lambda, find the user effects
predicted_ratings_reg <- validation %>% 
  left_join(movie_avgs_reg, by='movieId') %>%
  left_join(user_avgs_reg, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% 
  .$pred
## Make our predicted ratings
model_3_rmse <- RMSE(validation$rating,predicted_ratings_reg)
model_3_rmse
rmse_table_val <- bind_rows(rmse_table_val,
                          tibble(Method="Regularized Movie and User Effect Model",  
                                     RMSE = model_3_rmse ))






