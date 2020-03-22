# Recommender Systems
Recommender systems try to capture patterns and similar behaviors, to help predict what else you might like.

Advantages:
- users get a broader exposure to many different products they might be interested in
- This exposure encourages users towards continual usage or purchase of their product.
- better experience for the user but it benefits the service provider, with increased potential revenue and better security for its customers.

Main types: main difference between each, can be summed up by the type of statement that a consumer might make
- Content-based: “Show me more of the same of what I've liked before." 
    Content-based systems try to figure out what a user's favorite aspects of an item are, and then make recommendations on items that share those aspects.
- Collaborative filtering: “Tell me what's popular among my neighbors because I might like it too.” 
    Collaborative filtering techniques find similar groups of users, and provide recommendations based on similar tastes within that group.
- Hybrid recommender systems combining various mechanisms

Implementing recommender systems
- Memory-based:
  - we use the entire user-item dataset to generate a recommendation system
  - It uses statistical techniques to approximate users or items.
  - Examples: *Pearson Correlation, Cosine Similarity and Euclidean Distance*, among others.
- Model-based
  - a model of users is developed in an attempt to learn their preferences
  - Models can be created using Machine Learning techniques like regression, clustering, classification, and so on.

## Content-Based Recommender Systems

Try to recommed content based on their profiles, user ratings, etc. 
Content: items category, tags, genre, etc.

1. Create user profile
  - *Input user ratings*
  - *Item matrix*: Encode item with "One Hot Encoding” approach (Genre of movies are used here as a feature set.)
  - *Weighted Genre Matrix* = *Input user ratings* x *Item matrix*
  - *User profile*: aggregate the weighted genres and normalize the results
2. Figure out which item is most suited to be recommended to the user
  - Multiply *User profile matrix* x *Item candidate matrix* = *Weighted Item candidate matrix*
  - *Recommendation matrix* = aggregating *Weighted Item candidate matrix* 

Efficient method but in some cases it doesn't work. For example, use will get only recommendations with data based in their current profile. This problem can be solved by using other algorithms such as *colaborative filtering*.

Advantages and Disadvantages of Content-Based Filtering
- Advantages
  - Learns user's preferences
  - Highly personalized for the user
- Disadvantages
  - Doesn't take into account what others think of the item, so low quality item recommendations might happen
  - Extracting data is not always intuitive
  - Determining what characteristics of the item the user dislikes or likes is not always obvious

## Collaborative Filtering
