from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
import pandas as pd
from models.sensor import Sensor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class Predictor(LegacyBaseModel):
    def __init__(self, task=None, sensor=None):
        self.task = task
        self.sensor = sensor

    def train_it(self, non_norm=False, abs_val=False):
        dom_task = self.task
        nondom_sensor = Sensor.where(name=dom_task.get_opposite_sensor(self.sensor.name))[0]
        nondom_task = dom_task.get_counterpart_task()[0]

        dom_gs = dom_task.get_gradient_sets_for_sensor(self.sensor)
        nondom_gs = nondom_task.get_gradient_sets_for_sensor(nondom_sensor)

        dom_dfs = []
        nondom_dfs = []
        selected_features = ['grad_data__sum_values','grad_data__abs_energy','grad_data__mean_abs_change', 'grad_data__mean_change', 'grad_data__mean_second_derivative_central', 'grad_data__variation_coefficient','grad_data__standard_deviation','grad_data__skewness','grad_data__kurtosis','grad_data__variance','grad_data__root_mean_square','grad_data__mean']
        non_norm_features_dom = []
        non_norm_features_non_dom = []
        for loc in selected_features:
            non_norm_features_dom.append(loc.replace('grad_data', self.sensor.name))
            non_norm_features_non_dom.append(loc.replace('grad_data', nondom_sensor.name))


        for gs in dom_gs:
            try:
                if non_norm is False:
                    df_temp = pd.DataFrame(gs.get_aggregate_stats()['mean'][selected_features]).T
                else:
                    df_temp = pd.DataFrame(gs.get_aggregate_non_norm_stats(abs_val=abs_val)['mean'][non_norm_features_dom]).T
                    df_temp.columns = ['grad_data__' + col.split('__')[1] for col in df_temp.columns]
                df_temp['is_dominant'] = True  # dominant class
                dom_dfs.append(df_temp)
            except KeyError as e:
                print("Error with gradient set in dom_gs:", e)

        for gs in nondom_gs:
            try:
                if non_norm is False:
                    df_temp = pd.DataFrame(gs.get_aggregate_stats()['mean'][selected_features]).T
                else:
                    df_temp = pd.DataFrame(gs.get_aggregate_non_norm_stats(abs_val=abs_val)['mean'][non_norm_features_non_dom]).T
                    df_temp.columns = ['grad_data__' + col.split('__')[1] for col in df_temp.columns]
                df_temp['is_dominant'] = False  # non-dominant class
                nondom_dfs.append(df_temp)
            except KeyError as e:
                print("Error with gradient set in nondom_gs:", e)
        # concatenate both dataframes
        df = pd.concat(dom_dfs + nondom_dfs)

        accuracies = {}
        # Iterate over each feature
        for feature in selected_features:
            # select the feature and the target column
            X = df[[feature]]
            y = df['is_dominant']

            # split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # train a random forest classifier
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # make predictions on the test set
            y_pred = model.predict(X_test)

            # evaluate the model and store the accuracy for the feature
            accuracies[feature] = accuracy_score(y_test, y_pred)

        # Print the accuracy for each feature
        for feature, accuracy in accuracies.items():
            print(f'Accuracy for {feature}:', accuracy)

        return accuracies


    def train_it_nn(self, non_norm=False, abs_val=False):
            dom_task = self.task
            dom_sensors = Sensor.where(side='right')

            all_dfs = []
            selected_features = ['grad_data__sum_values','grad_data__abs_energy','grad_data__mean_abs_change', 'grad_data__mean_change', 'grad_data__mean_second_derivative_central', 'grad_data__variation_coefficient','grad_data__standard_deviation','grad_data__skewness','grad_data__kurtosis','grad_data__variance','grad_data__root_mean_square','grad_data__mean']
            nondom_task = dom_task.get_counterpart_task()[0]
            for sensor in dom_sensors:
                dom_gs = dom_task.get_gradient_sets_for_sensor(sensor)
                nondom_gs = nondom_task.get_gradient_sets_for_sensor(sensor)

                dom_dfs = []
                nondom_dfs = []
                
                non_norm_features = []
                for loc in selected_features:
                    non_norm_features.append(loc.replace('grad_data', sensor.name))

                for gs in dom_gs:
                    try:
                        if non_norm is False:
                            df_temp = pd.DataFrame(gs.get_aggregate_stats()['mean'][selected_features]).T
                        else:
                            df_temp = pd.DataFrame(gs.get_aggregate_non_norm_stats(abs_val=abs_val)['mean'][non_norm_features]).T
                            df_temp.columns = ['grad_data__' + col.split('__')[1] for col in df_temp.columns]
                        df_temp['is_dominant'] = True  # dominant class
                        dom_dfs.append(df_temp)
                    except KeyError as e:
                        print("Error with gradient set in dom_gs:", e)

                for gs in nondom_gs:
                    try:
                        if non_norm is False:
                            df_temp = pd.DataFrame(gs.get_aggregate_stats()['mean'][selected_features]).T
                        else:
                            df_temp = pd.DataFrame(gs.get_aggregate_non_norm_stats(abs_val=abs_val)['mean'][non_norm_features]).T
                            df_temp.columns = ['grad_data__' + col.split('__')[1] for col in df_temp.columns]
                        df_temp['is_dominant'] = False  # non-dominant class
                        nondom_dfs.append(df_temp)
                    except KeyError as e:
                        print("Error with gradient set in nondom_gs:", e)
                
                all_dfs += dom_dfs
                all_dfs += nondom_dfs
                df = pd.concat(all_dfs)
                
                X = df.drop('is_dominant', axis=1)
                y = df['is_dominant']

                # split data into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # Instantiate the model
                model = Sequential()
                model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
                model.add(Dense(64, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))

                # Compile the model
                model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

                # Train the model
                model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

                # Make predictions on the test set
                y_pred = (model.predict(X_test) > 0.5).astype("int32")

                # Evaluate the model
                accuracy = accuracy_score(y_test, y_pred)
                print('Accuracy:', accuracy)

                # return the trained model
                return model