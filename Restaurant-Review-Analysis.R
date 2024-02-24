library(data.table)
library(tidyverse)
library(text2vec)
library(data.table)
library(caTools)
library(glmnet)
library(inspectdf)
library(skimr)



nlp <- fread('nlpdata.csv')
nlp %>% view()

lapply(nlp,class)

nlp %>% inspect_na()
nlp %>% glimpse()
colnames(nlp)
nlp %>% dim()



#split data
set.seed(123)
split <- nlp$Liked %>% sample.split(SplitRatio = 0.8)
train <- nlp %>% subset(split == T)
test <- nlp %>% subset(split == F)



it_train <- train$Review %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$V1,
         progressbar = F) 


vocab <- it_train %>% create_vocabulary()

vocab %>% 
  arrange(desc(term_count)) %>% 
  head(150) %>% 
  tail(10) 


vectorizer <- vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(vectorizer)



glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,
            maxit = 1000)


glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")


it_test <- test$Review %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$V1,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)



preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$Liked, preds) %>% round(2)


#-----Removing stopwords and checking its effect on the result


stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is","was", "are")



vocab <- it_train %>% create_vocabulary(stopwords = stop_words)


pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5,
                   doc_proportion_min = 0.001)

pruned_vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10) 


vectorizer <- pruned_vocab %>% vocab_vectorizer()



dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 5,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$Liked, preds) %>% round(2)

#we can check ngrams effect on the result

vocab <- it_train %>% create_vocabulary(ngram = c(1L, 4L))

vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5)

ngram_vectorizer <- vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(ngram_vectorizer)
dtm_train %>% dim()

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,
            maxit = 850)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

dtm_test <- it_test %>% create_dtm(ngram_vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]

glmnet:::auc(test$Liked, preds) %>% round(2)



