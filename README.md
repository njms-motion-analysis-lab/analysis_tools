<img width="579" alt="Pasted Graphic 2" src="https://user-images.githubusercontent.com/15318220/224393092-5babe62d-46e0-45cd-823e-75b58c1d3688.png">



# Installation

  ### Remote
  * [VPN Access](https://it.rutgers.edu/virtual-private-network/)
  * [Amarel Access](https://oarc.rutgers.edu/resources/amarel)
  * Python and package manager (See Amarel docks above for installation instructions)

  ### Local
  * Set up [command line tools](https://www.freecodecamp.org/news/install-xcode-command-line-tools/)
  * Download text editor
  * Get [homebrew package manager](https://brew.sh)
  * Install python python3 from home-brew with `brew install python3`
  * Install python packages `pip install -r requirements.txt`
  * If you encounter an errror, try installing an indivudal package manually without specifying the version

# Setting up a Development Environment (some stuff I found online)


[stack-defn]: https://en.wikipedia.org/wiki/Solution_stack


## Phase 0: 

Here we will install basic developer tools, such as [homebrew][homebrew] (a 3rd party package manager for MacOS), Xcode (a library of developer tools provided by Apple), git (a version control system we will be using throughout the course), and Atom (a full-featured text-editor).


### Xcode

Let's start with Xcode. The Xcode command line tools are a requirement for installing the homebrew package manager in the next step. 


Install the Xcode command line tools by running the following from the console.

```sh
$ xcode-select --install
```

To conclude the installation you will need to agree to the Xcode license. Start the Xcode app, click "Agree", and allow the installation to finish. Then you can go ahead and quit the Xcode app.

### Homebrew
Homebrew is kind of like a low-tech App Store. It allows us access to and the ability to install a wide variety of software and command line tools from the console. These are distinct from those hosted on the App Store and will need to be managed by Homebrew.

Enter the following in your terminal to download and install Homebrew:

```sh
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

You will be given a list of dependencies that will be installed and prompted to continue or abort. Press `RETURN` to continue.

Let's break this command down a bit. `curl`, a command-line tool commonly used for downloading files from the internet, is used to download the Homebrew installation file. The `"$(...)"` transforms the file content into a string. Finally, the string is passed to our Ruby executable (`/usr/bin/ruby` is where this the system Ruby executable file is stored on our machine) with the `-e` flag to tell Ruby to run the argument as code. 

Check out the [Homebrew website][homebrew] to learn the basic commands.

[xcode]: https://itunes.apple.com/us/app/xcode/id497799835
[homebrew]: https://brew.sh/
[chrome-dl]: https://www.google.com/chrome/browser/desktop/index.html
[aa-chrome-tab]: https://github.com/appacademy/app-academy-chrome-tab
[linux-git]: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
[linux-rbenv]: https://github.com/rbenv/rbenv#basic-github-checkout

### Git
Git is a version control system that allows us to track, commit and revert changes to files within a directory. Here we will install it and add global user info.

```sh
# install git
brew install git

# makes git terminal output pretty
git config --global color.ui true

# this will mark you as the 'author' of each committed change
git config --global user.name "your name here"

# use the email associated with your GitHub account
git config --global user.email your_email_here
```

# Data
 
 ## Ingestion
   * The current data involves sensor coordinates while patients or healthy controls performed ADLs; it can be found on the lab motion analysis [Google Drive](https://drive.google.com/drive/u/0/folders/1bBezhJiFSPTo9fLG1i1jP0qqsuWlGBCj)
   * After downloading the folder of choice, move it to the `/raw_data` folder within this repo
 
 ## Database
   * To create the SQL tables, use the following commands:
     `from migrations.legacy_table import Table`
     `Table.create_tables()`
     `Table.update_tables()`
 
 ## Models
   * Running the command `python3 legacy_importer.py` will ingest npy files in the `raw_data` folder and create objects according to the schemas in the SQL database.
   * See the picture below for a rough diagram (TODO)
   * Additionally, this will break up motion for each sensor and trial into gradients (velocity) and subgradients based on zero value crossing. We also normalize subgradients gradients by length and store those.
   * From here, we use the `ts_fresh` package to extract features from sub gradients, normalized sub gradients, and the absolute value of sub gradients. Currently we are getting a comprehensive set of features. (TODO, show features and param serialization)
   * Running `python3 progress.py` will give a summary of each patient, as well as the tasks they completed and the trials measured and gradients captured.
 
 ## Training
 * As of now, most entrypoints for training models are in `mini_console.py` this file imports many of the models. By adding debuggers, train as well as debug the activities and models of choice.
 
 * Training classes are stored in the /prediction_tools folder and include the following 
   1. `MultiPredictor`--parent class for Predictor, has methods for displaying aggregate stats and training entry points.
   2. `Predictor`--set of training models for a given sensor_id and cohort_id. Predictor classes gather a patient cohort as well as params for normalization and abs value to train on relevent sensor data. By default this trains all models.
   3. `PredictorScore`--generated by Predictor model and contains the SHAP values for its parent predictor. This model also contains logic for generating heatmaps.
 
 **Guide for training**
 
   1. Find the set of patients and sensors by finding or creating a MultiPredictor object. For example:
       * to fetch results of block tasks:
         >mpa = MultiPredictor.where(cohort_id=1, task_id=3)[0]
       * you can view attributes of the object with the `mpa.attrs()`
       * you can view stats, like accuracy with `mpa.get_acc()`
       * you can view prior training runs with `mpa.get_all_preds()`
        >(Pdb) pp len(mpa.get_all_preds())
        >30
       * Since we have 10 sensors, this (likely) means we have a set of predictor for regular   stats, normalized stats, and absolute value stats.
       * If no predictors are present we can call [gen_scores_for_sensor](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_multi_predictor.py#L217-L226) on the MultiPredictor. By default this gets the 10 sensors shown at the top of the file and generates Predictor models for each.
 
   2. Call `.train_from()` or, if retraining an existing predictor, `retrain_from()`. [link]( https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L412-L429)
   3. This method first fetches a dataframe representing ts_fresh statistics for that sensor location [link]( https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L470-L513)
   4. This also generates dataframes representing ts fresh statistics from the coordinate non dominant task and sensor (i.e. right wrist dom => left wrist non dom)
   5. The `generate_dataframe` method has logic to manually switch for sensors for left hand dominant patients [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L337-L339)
   6. After getting the dataframe, we have logic to shrink the combined dataframe to roughly 50 x 50 (25 patients x 2 dominance levels by 50 selected features). [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L568)
   7. After fetching and triming the dataframe, we call `fit_multi_models`, [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L636). This method contains the training logic. It works in a few parts.
   8. Fetches classifiers and classifier hyperparams based on sample size (currently hardcoded to a smaller set of params for CP patients), [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L639-L643)
   9. Iterates through available classifiers and calls `train_classifier` on each.
```
grid_search = GridSearchCV(pipe, param_grid_classifier, cv=splitter, scoring='accuracy')
            grid_search.fit(X_train, y_train)
                    # Store best parameters and score
            train_accuracy = grid_search.score(X_train, y_train)
            best_classifier = grid_search.best_estimator_.named_steps['classifier']
            current_best_score = grid_search.best_score_
            current_best_params = grid_search.best_params_

            print(
                "Training accuracy...", train_accuracy, 
                "Current best training score:", current_best_score, 
                "Current best training params:", current_best_params,
                "Current best training classifier", best_classifier
            )
            # Predict on test set and calculate metrics
            y_pred = best_classifier.predict(X_test)
            y_pred_proba = best_classifier.predict_proba(X_test)[:, 1] if hasattr(best_classifier, "predict_proba")
```

  10. More on `train_classifeir` [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L917-L928)
  11. Selects splitter--K fold by defualt, LeeaveOneOut for small samples sizes, [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L929-L936)
  12. Ensures that test set has an even sample size, [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L926-L927)
  13. Drops is_dominant and patient id from training and column, [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L929-L936)
  14. Uses grid search to score splits selected with K-fold and parameters defined earlier. [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L938)
  15. Gets best parameters from training set, [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L939-L944)
  16. Gets accuracy scores from test set based on best params found in training set, [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L954)
  17. Gathers feature importances if available on classifier type.
  18. Gathers SHAP values if available on classifeir type.
  19. Gathers accuracy, precision, recall, f1, and AUC-ROC scores, [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L980-L988)
  20. Exits out of the fold loop and stores the average scores for each fold, [link](https://vscode.dev/github/stephen3292/motion_analysis/blob/6629205fcf92c9e3849ae983b159baeaaaee8639/prediction_tools/legacy_predictor.py#L1059-L1065)
  21. Prints and saves accuracy scores from training method above.
