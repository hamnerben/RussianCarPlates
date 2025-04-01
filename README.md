basic summary: predicting the price of Russian plates
- each one is 6 characters and then a region/government code
- submission scores closer to 0 are better
- predicting a higher cost than the actual is better than under predicting

feature engineering ideas:
- are particular regions more expensive?
- do certain letters or combos tend to cost more?

simple baseline: maybe train it on the mean price of all of them

we must try at least one deep learning method (use pytorch, maybe RNN or something? idk, look at hw 4 to see if it helps)

Nathan recommends random forest, he thinks it would respond well to decision trees but it might have a lot of columns
  - he also mentioned auto encoders but we haven't covered those yet
