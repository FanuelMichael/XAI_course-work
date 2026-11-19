import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import shap
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng(42)

#================================================================================================================
class SHAPExplainer:
    def __init__(self, model, feature_names, X_train, X_test,background_samples=50,algorithm=""):
        self.model = model
        self.feature_names = feature_names

        # Convert 3D → 2D
        self.X_train_2d = self._to_2d(X_train)
        self.X_test_2d = self._to_2d(X_test)

        # 2. Subsample background for Gradient/Kernel explainer
        n_background = min(background_samples, len(self.X_train_2d))
        background_idx = np.random.choice(len(self.X_train_2d), n_background, replace=False)
        self.X_background = self.X_train_2d[background_idx]

        # 1. Choose faster explainer explicitly
        self.algorithm = algorithm
        self.explainer = self._create_explainer()

        # Compute SHAP values only for the (smaller) test subset
        self.shap_exp = self.explainer(self.X_test_2d)
        self.shap_vals = self.shap_exp.values
        self.expected_value = self.shap_exp.base_values

        # GradientExplainer sometimes returns a list (for multi-class) or 3D array
        if isinstance(self.shap_vals, list):
            self.shap_vals = self.shap_vals[0]  # Take the first class (Churn)
        if len(self.shap_vals.shape) == 3:
            self.shap_vals = self.shap_vals.squeeze(-1)  # Converts (N, F, 1) to (N, F)

        # --- Handle None base values ---
        if self.expected_value is None or (isinstance(self.expected_value, np.ndarray) and self.expected_value.dtype == object):
            # Calculate the mean prediction on the background data manually
            bg_preds = self.model.predict(self.X_background, verbose=0).ravel()
            self.expected_value = np.mean(bg_preds)

    def _to_2d(self, arr):
        return arr.reshape(arr.shape[0], arr.shape[2]) if arr.ndim == 3 else arr

    def _predict_fn(self, x):
        if x.ndim == 2 and len(self.model.input_shape) == 3:
            x3 = x.reshape(x.shape[0], 1, x.shape[1])
            return self.model.predict(x3, verbose=0).ravel()
        return self.model.predict(x, verbose=0).ravel()

    def _create_explainer(self):
        if self.algorithm == "gradient":
            # GradientExplainer (uses background samples, much faster than Kernel)
            return shap.GradientExplainer(self.model,self.X_background,batch_size=256)
        elif self.algorithm == "permutation":
            # 4. Permutation explainer - very fast, approximate
            return shap.Explainer(self._predict_fn, self.X_background, algorithm="permutation")
        else:  # 'auto'
            return shap.Explainer(self._predict_fn, masker=shap.maskers.Independent(self.X_background),algorithm="auto")

    def get_top_shap_features(self, top_n=None):
        shap_abs_mean = np.mean(np.abs(self.shap_vals), axis=0)
        # Ensure we are sorting a 1D array
        if len(shap_abs_mean.shape) > 1:
            shap_abs_mean = shap_abs_mean.squeeze()
        sorted_idx = np.argsort(shap_abs_mean)[::-1].flatten()  # <-- Added .flatten()
        ordered_features = [self.feature_names[i] for i in sorted_idx]
        if top_n is not None:
            ordered_features = ordered_features[:top_n]
        return ordered_features

    # ---------------- shap summary plot-------------------
    def plot_summary(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(self.shap_vals,self.X_test_2d,feature_names=self.feature_names,
            max_display=12,show=False,rng=rng)
        ax.set_title("SHAP Summary (Global Importance)")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    # 1. SHAP Waterfall Plot------
    def plot_waterfall(self, cust_actual_id='', sample_idx=0):
        plt.figure(figsize=(10, 7))

        # --- NEW FIX: Construct a 1D Explanation object ---
        # GradientExplainer returns 3D values, so we use the squeezed arrays we created in __init__
        base_val = self.expected_value[sample_idx] if isinstance(self.expected_value,
                                                                 np.ndarray) else self.expected_value

        exp = shap.Explanation(values=self.shap_vals[sample_idx],base_values=base_val,
                               data=self.X_test_2d[sample_idx],feature_names=self.feature_names)

        shap.plots.waterfall(exp, show=False)

        # Center align the plot
        plt.gca().set_position([0.3, 0.1, 0.6, 0.8])  # [left, bottom, width, height]
        plt.title(f"SHAP Waterfall Plot (customer id: {cust_actual_id})")
        fig = plt.gcf()
        plt.close(fig)
        return fig

    # 2. SHAP force plot-----
    def shap_force_plot(self, cust_actual_id='', sample_idx=0):
        base_val = self.expected_value if not isinstance(self.expected_value, np.ndarray) \
                                       else self.expected_value[sample_idx]
        exp = shap.Explanation(values=self.shap_vals[sample_idx],base_values=base_val,
                               data=self.X_test_2d[sample_idx],feature_names=self.feature_names)

        shap.plots.force(exp, matplotlib=True, show=False, figsize=(24, 6))

        fig = plt.gcf()
        fig.set_dpi(300)
        plt.title(f"SHAP Force Plot for Customer Id: {cust_actual_id}", pad=60, fontsize=18)
        plt.subplots_adjust(top=0.75, bottom=0.15, left=0.05, right=0.95)
        plt.close(fig)
        return fig

    # 3. SHAP Decision Plot
    def shap_decision_plot(self,cust_actual_id='', sample_idx=0):
        self.shap_exp.feature_names = self.feature_names
        plt.figure(figsize=(10, 7))

        base_val = self.expected_value if not isinstance(self.expected_value, np.ndarray) else self.expected_value[
            sample_idx]

        shap.decision_plot(base_value=base_val,shap_values=self.shap_vals[sample_idx],
                features=self.X_test_2d[sample_idx],feature_names=self.feature_names,show=False)
        ax = plt.gca()

        # rotate labels 30 degree
        for label in ax.get_yticklabels():
            label.set_rotation(30)
            label.set_ha('right')

        # Get current position
        current_pos = ax.get_position()
        centered_left = (1 - current_pos.width) / 2  # Center calculation
        centered_left = max(0.05, min(centered_left, 0.4))  # Keep within bounds
        # Apply centered position
        ax.set_position([centered_left+0.03,current_pos.y0,current_pos.width,current_pos.height])
        plt.title(f"SHAP Decision Plot for Cstomer Id: {cust_actual_id})")
        fig = plt.gcf()
        plt.close(fig)
        return fig