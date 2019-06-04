library(tidyverse) # metapackage with lots of helpful functions
library(tidytext) # working with text
library(keras) # deep learning with keras

suppressMessages(library(stringr))
suppressMessages(library(DT))
suppressMessages(library(igraph))
suppressMessages(library(ggraph))
suppressMessages(library(tm))
suppressMessages(library(wordcloud2))
suppressMessages(library(wordcloud))
suppressMessages(library(caret))
suppressMessages(library(dplyr))
suppressMessages(library(magrittr))
suppressMessages(library(plyr))
fillColor = "#FFA07A"
fillColor2 = "#F1C40F"
suppressMessages(library(tidyverse)) # general utility & workflow functions
suppressMessages(library(tidytext)) # tidy implimentation of NLP methods
suppressMessages(library(topicmodels)) # for LDA topic modelling
suppressMessages(library(tm)) # general text mining functions, making document term matrixes
suppressMessages(library(SnowballC)) # for stemming
suppressMessages(library(glue)) ##for pasting strings



# Reading the dataset into a data frame
train = read.csv(file = 'gs://gentle-dominion-233820/r-cloudml/bbc-text.csv', stringsAsFactors = FALSE)  #set the path to the csv file
dim(train)
train %>% head()


#finding missing values
for(i in 1:ncol(train)) {
  colName <- colnames(train[i])
  pctNull <- sum(is.na(train[,i]))/length(train[,i])
  if (pctNull > 0.90) {
    print(paste("Column ", colName, " has ", round(pctNull*100, 3), "% of nulls"))
  }
    else {
        print("No Null values")
    }
}

#Exploratory Data analysis

cat("Basic preprocessing & stats of Phrase...\n")
train <- train %>%
  mutate(length = str_length(text),
         ncap = str_count(text, "[A-Z]"),
         ncap_len = ncap / length,
         nexcl = str_count(text, fixed("!")),
         nquest = str_count(text, fixed("?")),
         npunct = str_count(text, "[[:punct:]]"),
         nword = str_count(text, "\\w+"),
         nsymb = str_count(text, "&|@|#|\\$|%|\\*|\\^"),
         nsmile = str_count(text, "((?::|;|=)(?:-)?(?:\\)|D|P))"))

head(train,2)




train_Tokens <- train %>% unnest_tokens(word, text)
head(train_Tokens,3)






train %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word) %>%
  dplyr:: count(word,sort = TRUE) %>%
  ungroup() %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>%
  head(20) %>%
  ggplot(aes(x = word,y = n)) +
  geom_bar(stat='identity',colour="white", fill =fillColor) +
  geom_text(aes(x = word, y = 1, label = paste0("(",n,")",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Word', y = 'Word Count',
       title = 'Top 20 most Common Words') +
  coord_flip() +
  theme_bw()





train %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word) %>%
  #   filter(!word %in% custom_stop_words$word) %>%
  dplyr::count(word,sort = TRUE) %>%
  ungroup()  %>%
  head(50) %>%

  with(wordcloud(word, n, max.words = 50,colors=brewer.pal(8, "Dark2")))





# Find top categories
Topcategories = train %>%
  group_by(category) %>%
  tally(sort = TRUE)
Topcategories





contributions <- train %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word) %>%
  #   filter(!word %in% custom_stop_words$word) %>%
  dplyr::count(category, word, sort = TRUE) %>%
  ungroup() %>%

  inner_join(get_sentiments("bing"), by = "word") %>%
  group_by(word) %>%
  dplyr::summarize(occurences = n(),
                   contribution = sum(n))

contributions %>%
  top_n(20, abs(contribution)) %>%
  mutate(word = reorder(word, contribution)) %>%
  ggplot(aes(word, contribution, fill = contribution > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() + theme_bw()





contributions <- train%>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word) %>%
  #   filter(!word %in% custom_stop_words$word) %>%
  dplyr::count(category, word, sort = TRUE) %>%
  ungroup() %>%

  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(word) %>%
  dplyr::summarize(occurences = n(),
                   contribution = sum(score))

contributions %>%
  top_n(20, abs(contribution)) %>%
  mutate(word = reorder(word, contribution)) %>%
  ggplot(aes(word, contribution, fill = contribution > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() + theme_bw()





#Changing categorial data to numerical data
train[,1] <- sapply(train[,1],switch,"business"=0,"tech"=1,"sport"=2, "entertainment" = 3, "politics" =4)
train %>% head()

classer = c("business","tech","sport","entertainment","politics")




# Setup some parameters

max_words <- 10000 # Maximum number of words to consider as features
maxlen <- 64 # Text cutoff after n words


library(dplyr)
# Prepare to tokenize the text

full <- rbind(train %>% select(text))
texts <- full$text

tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(texts)

# Tokenize - i.e. convert text into a sequence of integers

sequences <- texts_to_sequences(tokenizer, texts)
word_index <- tokenizer$word_index

# Pad out texts so everything is the same length

data = pad_sequences(sequences, maxlen = maxlen)



dim(data)





# Split back into train and test

train_matrix = data[1:nrow(train),]
# labels = c(train$category)
labels <-as.numeric(train$category)
# labels <- rbind(train %>% select(category))

one_hot_labels <- to_categorical(labels,num_classes = 5)
# Prepare a validation set

set.seed(1337)
training_samples = nrow(train_matrix)*0.80
test_samples = nrow(train_matrix)*0.20

indices = sample(1:nrow(train_matrix))
training_indices = indices[1:training_samples]
test_indices = indices[(training_samples + 1): (training_samples + test_samples)]

x_train = train_matrix[training_indices,]
y_train = one_hot_labels[training_indices,]

x_test = train_matrix[test_indices,]
y_test = one_hot_labels[test_indices,]
dim(x_train)
y_train
dim(y_train)



#Reading the wiki-news-embedding matrix 
lines <- readLines('gs://gentle-dominion-233820/r-cloudml/wiki-news-300d-1M.vec')  #set the path to the vetor file

#creating environment variable
news_index = new.env(hash = TRUE, parent = emptyenv())

lines <- lines[2:length(lines)]

pb <- txtProgressBar(min = 0, max = length(lines), style = 3)
for (i in 1:length(lines)){
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word<- values[[1]]
  news_index[[word]] = as.double(values[-1])
  setTxtProgressBar(pb, i)
}

# Create our embedding matrix

news_dim = 300
news_matrix = array(0, c(max_words, news_dim))

for (word in names(word_index)){
  index <- word_index[[word]]
  if (index < max_words){
    news_vector = news_index[[word]]
    if (!is.null(news_vector))
      news_matrix[index+1,] <- news_vector # Words without an embedding are all zeros
  }
}



input <- layer_input(
  shape = list(NULL),
  dtype = "int32",
  name = "input"
) 

# Model layers

embedding <- input %>% 
    layer_embedding(input_dim = max_words, output_dim = news_dim, name = "embedding")

lstm <- embedding %>% 
    layer_lstm(units = maxlen,dropout = 0.25, recurrent_dropout = 0.25, return_sequences = FALSE, name = "lstm")

dense <- lstm %>%
    layer_dense(units = 128, activation = "relu", name = "dense") 

# dense <- lstm %>%
#     layer_dense(units = 64, activation = "relu", name = "dense") 

predictions <- dense %>% 
    layer_dense(units = 5, activation = "softmax")
    

# fitting the model

model <- keras_model(input, predictions)

# Freezing the weights in the model

get_layer(model, name = "embedding") %>% 
  set_weights(list(news_matrix)) %>% 
  freeze_weights()

metric_mean_pred <- custom_metric("mean_pred", function(y_true, y_pred) {
  k_mean(y_pred)
})

# Compiling everything together

model %>% compile(
  optimizer = optimizer_adam(),
  loss = "categorical_crossentropy",
  metrics = c("accuracy","mean_squared_error","msle")
)


# Printing the entire architecture 

print(model)




# Train model 

history <- model %>% fit(
  x_train,
  y_train,
  batch_size = 32,
  validation_data = list(x_test, y_test),
  epochs = 20,
  view_metrics = TRUE,
  verbose = 0
)



print(history)
plot(history)


model %>% evaluate(x_test, y_test,verbose = 1)

predictions = model %>% predict(x_test,verbose=1,batch_size=32)

dim(predictions)
a <- floor(runif(50, min=1, max=445))


for(i in 25:50){
  maxi=max(predictions[i,])
  print(train[test_indices[i],])
  for(j in 1:5){
    if(y_val[i,j]==1)
      print(classer[j])
    if(predictions[i,j]==maxi)
      print(classer[j])
  }
    
}

