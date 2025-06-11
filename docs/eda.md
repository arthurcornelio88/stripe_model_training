## 📊 EDA Summary — Automatic Fraud Detection

### 1. **Class Imbalance**

![is\_fraud](../data/img/isfraud.png)

* Only **\~0.39%** of transactions are fraudulent.
* ➤ This confirms the need to:

  * Handle imbalance using `class_weights` or `scale_pos_weight` (for CatBoost),
  * Focus on metrics like **F1-score** or **AUC**, not accuracy.

---

### 2. **Impact of Transaction Category**

![fraud\_par\_cat](../data/img/fraud_par_cat.png)

* Highest fraud risk categories:

  * `shopping_net`, `misc_net`, `grocery_pos`
* ➤ Highly discriminative feature — needs proper encoding.

---

### 3. **Transaction Amount by Category**

![montant\_par\_cat](../data/img/montant_par_cat.png)

* Several categories show highly skewed transaction amounts.
* ➤ Applying a `log1p` transformation on `amt` can help stabilize variance.

---

### 4. **Age and Fraudulent Behavior**

![fraud\_age](../data/img/fraud_age.png)

* Frauds are more common among users aged **30–55**.
* ➤ `age` is an informative feature — definitely worth keeping.

---

### 5. **Gender and Fraud**

![fraud\_genre](../data/img/fraud_genre.png)

* No significant difference in fraud rates between `M` and `F`.
* ➤ Still worth including, though likely low impact.

---

### 6. **Temporal Patterns**

![trans\_date](../data/img/trans_date.png)

* Fraud frequency varies over time, but shows no clear trend.
* ➤ Consider extracting time-based features like:

  * `dayofweek`, `hour`, `is_weekend`, etc.

---

### 7. **Geographical Distribution**

![output](../data/img/clients_merchants.png)

* Both customers and merchants are densely located around major U.S. cities.
* ➤ Suggests creating a **distance-based feature** between customer and merchant.

---

## 🔧 Next Step: `02_preprocessing.py`

Here’s a breakdown of preprocessing and feature engineering actions:

| Action                  | Variables                                        | Description                                               |
| ----------------------- | ------------------------------------------------ | --------------------------------------------------------- |
| 🧹 Drop columns         | `first`, `last`, `street`, `trans_num`           | Not useful for modeling                                   |
| 📅 Date features        | `trans_date_trans_time`, `dob`                   | Generate `age`, `hour`, `dayofweek`, `month`              |
| 📏 Distance calculation | `lat/long`, `merch_lat/merch_long`               | Use haversine or euclidean distance                       |
| 🔣 Encoding             | `gender`, `category`, `state`, `job`, `merchant` | Use target encoding or leverage CatBoost’s native support |
| 💰 Log transformation   | `amt`                                            | Apply `np.log1p(amt)` to reduce skewness                  |
| ⚖️ Imbalance handling   | `is_fraud`                                       | Do not undersample — handle via model parameters instead  |