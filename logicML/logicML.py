import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import seaborn as sns



class logicML:
    def __init__(self, train_df, test_df, target_var='Histology group', remove_var='FIT 1', max_models=20, seed=124):
        # Ensure test_df is a pandas DataFrame
        if not isinstance(test_df, pd.DataFrame):
            try:
                self.test_df = pd.DataFrame(test_df)
            except:
                raise ValueError("test_df could not be converted to a pandas DataFrame.")
        else:
            self.test_df = test_df
            
        self.train_df = train_df  # Initialize train_df as a class attribute
        self.target_var = target_var
        self.remove_var = remove_var
        self.max_models = max_models
        self.seed = seed
        self.initialize_h2o()
        self.target_fpr = 0.2  # Set the default target FPR
        self.target_tpr = 0.8  # Set the default target TPR

        
    def initialize_h2o(self):
        h2o.init()
        
    def prepare_data(self):
        X_train, y_train = self.train_df.iloc[:,:-1], self.train_df.iloc[:,-1]
        train_df = pd.concat([X_train, y_train], axis=1)
        train = h2o.H2OFrame(train_df)
        test = h2o.H2OFrame(self.test_df)
        
        y = self.target_var
        x = train.columns
        x.remove(y)
        x.remove(self.remove_var)
        
        train[y] = train[y].asfactor()
        test[y] = test[y].asfactor()
        
        return train, test, x, y

    def train_model(self, train, x, y):
        aml = H2OAutoML(max_models=self.max_models, seed=self.seed)
        aml.train(x=x, y=y, training_frame=train)
        return aml

    def evaluate_model(self, aml, test, x, y):
        lb = aml.leaderboard
        lb.head(rows=lb.nrows)
        best_model = aml.leader
        print(best_model)
        best_model.model_performance(test)
        
        preds = best_model.predict(test)
        preds = preds.as_data_frame()
        
        return preds

    def plot_roc_curve(self, y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Store fpr and tpr as class attributes
        self.fpr = fpr
        self.tpr = tpr

        plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
        
        return thresholds

    def find_best_threshold(self, fpr, tpr, thresholds, target_fpr, target_tpr):
        idx = np.argmin(np.abs(fpr - target_fpr) + np.abs(tpr - target_tpr))
        return thresholds[idx]

    def plot_multiple_roc_curves(self, y_true, pred1, pred2, y_pred_union):
        fpr1, tpr1, _ = roc_curve(y_true, pred1)
        roc_auc1 = auc(fpr1, tpr1)
        fpr2, tpr2, _ = roc_curve(y_true, pred2)
        roc_auc2 = auc(fpr2, tpr2)
        fpr_union, tpr_union, _ = roc_curve(y_true, y_pred_union)
        roc_auc_union = auc(fpr_union, tpr_union)
        plt.figure()
        plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='Model 1 ROC curve (area = %0.2f)' % roc_auc1)
        plt.plot(fpr2, tpr2, color='blue', lw=2, label='Model 2 ROC curve (area = %0.2f)' % roc_auc2)
        plt.plot(fpr_union, tpr_union, color='green', lw=2, label='Ensemble Model ROC curve (area = %0.2f)' % roc_auc_union)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()



    def ensemble_predictions(self, y_true, pred1, pred2, threshold1, threshold2, logic_gate='or'):
        y_pred1 = (pred1 > threshold1).astype(int)
        y_pred2 = (pred2 > threshold2).astype(int)
        
        if logic_gate == 'or':
            y_pred_union = np.logical_or(y_pred1, y_pred2)
        elif logic_gate == 'and':
            y_pred_union = np.logical_and(y_pred1, y_pred2)
        elif logic_gate == 'xor':
            y_pred_union = np.logical_xor(y_pred1, y_pred2)
        elif logic_gate == 'nand':
            y_pred_union = np.logical_not(np.logical_and(y_pred1, y_pred2))
        elif logic_gate == 'nor':
            y_pred_union = np.logical_not(np.logical_or(y_pred1, y_pred2))
        elif logic_gate == 'xnor':
            y_pred_union = np.logical_not(np.logical_xor(y_pred1, y_pred2))
        else:
            raise ValueError("Invalid logic gate specified. Use 'or', 'and', 'xor', 'nand', 'nor', or 'xnor'.")
        
        # Note that we removed the hard-coded 'or' operation from here
        accuracy1 = accuracy_score(y_true, y_pred1)
        accuracy2 = accuracy_score(y_true, y_pred2)
        accuracy_union = accuracy_score(y_true, y_pred_union)
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred_union).ravel()
        specificity = TN / (TN + FP)
        sensitivity = TP / (TP + FN)
        print(f"Accuracy of FIT model: {accuracy1 * 100:.2f}%")
        print(f"Accuracy of model 2: {accuracy2 * 100:.2f}%")
        print(f"Accuracy of ensemble model: {accuracy_union * 100:.2f}%")
        print(f"Specificity of ensemble model: {specificity * 100:.2f}%")
        print(f"Sensitivity of ensemble model: {sensitivity * 100:.2f}%")
        return y_pred_union, accuracy_union, specificity, sensitivity





    def get_variable_importance(self, best_model):
        """Get and plot the variable importance from the best model."""
        varimp = best_model.varimp()
        print(varimp)
        
        # Extract top 50 variables for visualization
        top = varimp[0:50]
        feature_names = [item[0] for item in top]
        importance_scores = [item[1] for item in top]

        plt.bar(feature_names, importance_scores)
        plt.xlabel('Feature')
        plt.ylabel('Importance Score')
        plt.title('Top 50 Feature Importance')
        plt.xticks(rotation=90)
        plt.show()

        return varimp

    def train_with_topn_features(self, varimp, n, train, target, test, test_df, y_pred1, tpr_target=None, fpr_target=None):
        """Train the model using the top N important features and visualize the result with an ROC curve."""
        top_n_var = [row[0] for row in varimp[:n]]
        col_indices = [train.names.index(var) for var in top_n_var]
        
        top_data = train[:, col_indices]
        top_data = top_data.cbind(train[target])
        
        aml_2 = H2OAutoML(max_models=10, seed=42)
        aml_2.train(x=top_data.columns[:-1], y=target, training_frame=top_data)

        preds_var10_2 = aml_2.leader.predict(test)
        preds_var10_2 = preds_var10_2.as_data_frame()
        
        # Get the ROC curve and find the best threshold
        fpr, tpr, thresholds = roc_curve(test_df[target], preds_var10_2['p1'])
        
        if tpr_target is None or fpr_target is None:
            best_threshold = thresholds[np.argmax(tpr - fpr)]
        else:
            best_threshold_idx = np.argmin(np.abs(tpr - tpr_target) + np.abs(fpr - fpr_target))
            best_threshold = thresholds[best_threshold_idx]
        
        y_v15_pred2 = (preds_var10_2['p1'] > best_threshold).astype(int)
        
        return y_v15_pred2, best_threshold

    def permutation_and_combination(self, varimp, n, m, train, target, test, test_df, y_pred1):
        """Perform combinations on the top N important features to find the best feature set, selecting M features to find the best combination at a time. (sort via AUC score)

        Args:
        varimp: list of tuples
            The variable importances obtained from a trained model.
        n: int
            The number of top features to consider for generating combinations.
        m: int
            The number of features to select in each combination.
        train: H2OFrame
            The training data.
        target: str
            The target variable name.
        test: H2OFrame
            The testing data.
        test_df: pandas DataFrame
            The testing data in pandas DataFrame format for easy manipulation in Python.
        y_pred1: pandas Series/1D array
            The predictions of a baseline model to combine with the predictions from the new model trained on the subset of features.

        Returns:
        results: list of tuples
            A list of feature combinations and their corresponding AUCs, sorted in descending order of AUC.
        """
        top_n_var = [row[0] for row in varimp[:n]]
        
        results = []
        combinations = itertools.combinations(top_n_var, m)
        
        for combination in combinations:
            col_indices = [train.names.index(var) for var in combination]
            top_data = train[:, col_indices]
            top_data = top_data.cbind(train[target])

            aml_2 = H2OAutoML(max_models=10, seed=42, exclude_algos=["GBM", "DeepLearning"])
            aml_2.train(x=top_data.columns[:-1], y=target, training_frame=top_data)
            
            y_v15_pred2 = aml_2.leader.predict(test).as_data_frame()['p1']
            y_pred_combined = (y_pred1 + y_v15_pred2) / 2
            
            fpr, tpr, thresholds = roc_curve(test_df[target], y_v15_pred2)
            best_threshold = thresholds[np.argmax(tpr - fpr)]
            
            y_pred_combined_best_threshold = (y_pred_combined > best_threshold).astype(int)
            auc = roc_auc_score(test_df[target], y_pred_combined_best_threshold)
            
            results.append((combination, auc))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        plt.figure(figsize=(10, 20))
        sns.barplot(x=[r[1] for r in results], y=[str(r[0]) for r in results])
        plt.xlabel('AUC')
        plt.ylabel('Feature Combinations')
        plt.title('Feature Combinations by AUC')
        plt.show()
        
        return results


    def roc_and_confusion_matrix(self, results, train, test, test_df, y_pred1, target='Histology group'):
        """Plot the ROC curve and confusion matrix for the best combination of features found."""
        best_combination, _ = max(results, key=lambda x: x[1])
        
        col_indices = [train.names.index(var) for var in best_combination]
        top_data = train[:, col_indices]
        top_data = top_data.cbind(train[target])

        aml_best = H2OAutoML(max_models=10, seed=42)
        aml_best.train(x=top_data.columns[:-1], y=target, training_frame=top_data)

        preds_best = aml_best.leader.predict(test).as_data_frame()['p1']
        fpr, tpr, _ = roc_curve(test_df[target], preds_best)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        # Add code to plot the confusion matrix using seaborn's heatmap function
        y_pred_best = (preds_best > 0.5).astype(int)  # Adjust the threshold as needed
        cm = confusion_matrix(test_df[target], y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_fit_manual_roc(self, target_fpr=None, target_tpr=None):
        """Plot the ROC curve for manual FIT predictions and determine the best threshold based on target FPR and TPR."""
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        import numpy as np

        y_true = self.train_df[self.target_var].values
        y_scores = self.train_df['FIT 1'].values

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for FIT Manual')
        plt.legend()
        plt.show()
        
        # If target FPR and TPR are not provided, use the default values
        if target_fpr is None:
            target_fpr = self.target_fpr
        if target_tpr is None:
            target_tpr = self.target_tpr

        idx = np.argmin(np.abs(fpr - target_fpr) + np.abs(tpr - target_tpr))
        self.fit_threshold = thresholds[idx]
        print(f"Selected Threshold: {self.fit_threshold}")


    def fit_manual_prediction(self):
        """Generate manual FIT predictions using the determined threshold."""
        self.test_df['FIT_manual_pred'] = (self.test_df['FIT 1'] > self.fit_threshold).astype(int)


    def stage_one(self):
        """
        Stage One: Data Preparation and ROC Curve Plot
        - Data preparation
        - Model training
        - Model evaluation
        - ROC curve plotting to visually aid in threshold determination
        """
        train, test, x, y = self.prepare_data()
        aml = self.train_model(train, x, y)
        preds = self.evaluate_model(aml, test, x, y)
        self.plot_fit_manual_roc()  # You can specify target_fpr and target_tpr here if known
        self.fit_manual_prediction()  # Generating FIT manual predictions using the threshold determined above
        thresholds = self.plot_roc_curve(self.test_df[self.target_var], preds['p1'])

        # Save necessary data for use in later stages
        self.train = train
        self.test = test
        self.x = x
        self.y = y
        self.aml = aml
        self.preds = preds
        self.thresholds = thresholds

    def stage_two(self, threshold='best'):
        """
        Stage Two: Feature Importance and Top N Feature Training
        - Variable importance determination
        - Training with top N important features
        - Accepts a threshold parameter with a default value of 'best', which uses the optimal threshold
        """
        if threshold == 'best':
            threshold = self.thresholds[np.argmax(self.tpr - self.fpr)]  # Now correctly gets the best threshold using self.tpr and self.fpr

        varimp = self.get_variable_importance(self.aml.leader)
        y_pred1, best_threshold = self.train_with_topn_features(varimp, 10, self.train, self.y, self.test, self.test_df, self.preds['p1'])

        # Save necessary data for use in the next stage
        self.varimp = varimp
        self.y_pred1 = y_pred1
        self.best_threshold = best_threshold

    def stage_three(self):
        """
        Stage Three: Permutation and Combination Analysis
        - Permutation and combination of top N important features to find the best feature set
        """
        results = self.permutation_and_combination(self.varimp, 10, self.train, self.y, self.test, self.test_df, self.y_pred1)
        self.roc_and_confusion_matrix(results, self.train, self.test, self.test_df, self.y_pred1, target=self.y)



    def execute_pipeline(self, threshold='best'):
        """
        Execute the full pipeline by sequentially executing all three stages
        with the possibility to specify the threshold for stage two.
        """
        self.stage_one()
        self.stage_two(threshold)
        self.stage_three()
