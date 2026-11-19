import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer

#======================================================================================================
class LIMEExplainer:
    def __init__(self, model, feature_names, X_train, X_test):

        self.model = model
        self.feature_names = feature_names
        # Convert 3D -> 2D for LIME
        self.X_train_2d = self.convert_to_2d(X_train)
        self.X_test_2d = self.convert_to_2d(X_test)

        # LIME explainer initialization
        self.explainer = LimeTabularExplainer(training_data=self.X_train_2d,feature_names=self.feature_names,
                         mode="classification",discretize_continuous=True)

    def convert_to_2d(self, arr):
        # Convert 3D(samples, 1, features) -> 2D(samples, features)
        return arr.reshape(arr.shape[0], arr.shape[2]) if arr.ndim == 3 else arr

    def _predict_fn(self, x):
        # x is 2D (batch, features)
        preds = self.model.predict(x)  # This returns a numpy array in eager mode
        # If it's still a tensor, force conversion:
        if hasattr(preds, 'numpy'):
            preds = preds.numpy()
        return np.column_stack([1 - preds, preds])

    def explain_sample(self, sample_idx=0, num_features=10):
        """Generate a LIME explanation for a single sample."""
        instance = self.X_test_2d[sample_idx]
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self._predict_fn,
            num_features=num_features)
        return explanation

    # 1. LIME Bar Chart plot
    def lime_bar_chart_plot(self, cust_actual_id='', sample_idx=0, num_features=10):
        """Plot a LIME local explanation as a bar chart."""
        explanation = self.explain_sample(sample_idx, num_features)
        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(8, 6)
        # --- Center the axes inside the figure ---
        ax = fig.axes[0]
        ax.set_anchor('C')  # 'C' = center

        # --- Modify bar colors: red for positive, blue for negative ---
        for patch in ax.patches:
            width = patch.get_width()  # For horizontal bars
            patch.set_color('red' if width > 0 else 'blue')

        plt.title(f"LIME Bar Chart of Customer Id: {cust_actual_id}")
        plt.tight_layout()
        plt.close(fig)
        return fig

    # 2. LIME Decision Path
    def lime_decision_path_plot(self, cust_actual_id='', sample_idx=0, num_features=10):
        # Visualizes the cumulative sum of contributions starting from the intercept
        exp = self.explain_sample(sample_idx, num_features)
        intercept = exp.intercept[1]
        local_weights = exp.as_list()

        features, values = zip(*local_weights)
        all_values = [intercept] + list(values)
        all_labels = ['Intercept (Baseline)'] + list(features)
        cumulative = np.cumsum(all_values)

        fig=plt.figure(figsize=(12, 6))
        plt.step(range(len(cumulative)), cumulative, where='mid', label='Cumulative Prediction Probability',
                 color='green')
        # --- Set bar colors: red for positive, blue for negative ---
        bar_colors = ['red' if v > 0 else 'blue' for v in all_values]
        plt.bar(range(len(all_values)), all_values, alpha=1.0, label='Feature Contribution', color=bar_colors)
        plt.xticks(range(len(all_labels)), all_labels, rotation=45, ha='right')
        plt.title(f"LIME Decision Path Plot of Customer Id: {cust_actual_id}")
        plt.axhline(y=0.5, color='red', linestyle='--', label='Churn Threshold (0.5)')
        plt.legend()
        plt.tight_layout()
        plt.close(fig)
        return fig

    # --- LIME Heat-Map Plot Method ---
    def lime_heatmap_plot(self, cust_actual_id='', sample_idx=0, num_features=10):
        exp = self.explain_sample(sample_idx, num_features)
        local_weights = exp.as_list()
        features, values = zip(*local_weights)

        # Reshape values into a 2D array (1, num_features) for heatmap visualization
        values_arr = np.array(values).reshape(1, -1)

        fig, ax = plt.subplots(figsize=(11, 3.5), dpi=200)
        sns.heatmap(values_arr,annot=True, fmt=".3f",cmap="vlag",center=0,xticklabels=features,
        yticklabels=["Weight"],cbar=True,ax=ax)
        plt.xticks(rotation=30, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.title(f"LIME Feature Contribution Heat-Map (Customer Id: {cust_actual_id})", fontsize=12,
                      fontweight='bold', pad=12)
        plt.tight_layout()
        plt.close(fig)
        return fig