put notes for report here

models:
    - used the mean and then the max as our baseline "model" (102.8921, 198.5034)
    - RNN ()
    - random forest 
        - initially focused on the regions, not much better than the baseline. 
        - Then focused on is_gov and region info, still not much better but small improvement
        - added in features like road advantage, significance. slight improvement
        - scaling the features didn't seem to help
        - added more features (region code, plate digits, plate length) and used grid search to find better hyperparams. this took way longer to run, but finally showed a significant improvement. also evaluated the importance of the features, so we could remove the unnecessary features
        - took out the gridsearch and it somehow was better? not sure how
        - added in the dates to the features, didnt help



notable features to maybe single out:
- is it a government vehicle
- which region is it from

-----------------------------------------------------------------------------
final report recs:
must follow ACM two-column conference template https://www.acm.org/publications/proceedings-template 

- need to have made at least 6 non dummy submissions
- report should be 3 to (at most) 6 pages, and structured like a small research paper
  - intro:
      - overview of project. What problem are we tackling?
      - summarize accomplishments, including scores and public/private leaderboard rankings if available
      - **write an independent paragraph about what you have learned**
  - approaches
      - what ML techniques did we try and why?
  - experiments
      - describe the experiments that we ran, the outcomes, and any error analysis that we did
      - at least two tables are required; one is to summarize the performance of different approaches, while another is to show the performance of the best models under hyperparameter tuning
  - summary and future work
      - if we had more time, how would we continue to work on the project?
  - reference
  - appendix (optional)
