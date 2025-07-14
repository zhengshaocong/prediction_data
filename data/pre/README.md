# /pre 目录说明与更换规则

本目录下的 `feature_blocks.py` 和 `clean_data.py` 是特征工程与数据清洗的核心模块，通常会随着训练集的变化而同步更换。请严格按照以下规则进行维护和扩展。

---

## 1. feature_blocks.py 维护与更换规则

- **用途**：定义所有特征工程函数，并通过 `FEATURE_BLOCKS` 控制启用哪些特征。
- **暴露接口**：只暴露 `default(df, train_df=None, feature_blocks=None)`，主流程只需 import 该函数。

### 新增特征函数
1. 在文件内新增特征工程函数，函数签名一般为：
   ```python
   def my_new_feature(df, train_df=None):
       # ...实现...
       return df
   ```
2. 在 `feature_block_funcs` 字典中注册：
   ```python
   feature_block_funcs = {
       # ...已有映射...
       'my_new_feature': my_new_feature,
   }
   ```
3. 在 `FEATURE_BLOCKS` 列表中添加对应特征名（按需调整顺序）：
   ```python
   FEATURE_BLOCKS = [
       # ...已有特征...
       'my_new_feature',
   ]
   ```

### 更换/适配新训练集
- 根据新训练集的字段和业务需求，增删或调整特征函数，并同步维护 `FEATURE_BLOCKS` 顺序。
- 只需保证 `default(df, ...)` 能正确处理新数据。

---

## 2. clean_data.py 维护与更换规则

- **用途**：定义所有数据清洗函数，并通过 `STEPS` 控制清洗流程。
- **暴露接口**：只暴露 `default(df)`，主流程只需 import 该函数。

### 新增清洗函数
1. 在文件内新增清洗函数，函数签名一般为：
   ```python
   def clean_my_rule(df):
       # ...实现...
       return df
   ```
2. 在 `STEP_FUNCS` 字典中注册：
   ```python
   STEP_FUNCS = {
       # ...已有映射...
       'my_rule': clean_my_rule,
   }
   ```
3. 在 `STEPS` 列表中添加对应步骤名（按需调整顺序）：
   ```python
   STEPS = [
       # ...已有步骤...
       'my_rule',
   ]
   ```

### 更换/适配新训练集
- 根据新训练集的异常情况和业务需求，增删或调整清洗函数，并同步维护 `STEPS` 顺序。
- 只需保证 `default(df)` 能正确处理新数据。

---

## 3. 其它说明
- 两个文件均无需对外暴露除 `default` 以外的内容。
- 只需在主流程中 import `default`，可自由命名。
- 更换数据集时，务必同步维护特征/清洗函数和控制列表（`FEATURE_BLOCKS`、`STEPS`）。 