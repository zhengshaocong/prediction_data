import os
import pandas as pd
from src import (
    preprocess_data, preprocess_features, save_model, load_model,
    RegressionTrainConfig, DataConfig, ModelConfig, TrainConfig, ParamTuningConfig,
    CrossValidateConfig, cross_validate_regression, load_config, get_config_value, ensure_directories,
    SegmentedRegressionConfig, segmented_regression, auto_bins
)
from src.models.utils.model_utils_tools import plot_feature_importance
import importlib.util
from src.utils.tools import check_and_get_feature_blocks

def run_pipeline(config_path='./config.yaml'):
    # 加载配置
    config = load_config(config_path)

    # 动态加载特征工程模块
    preprocess_module_path = config['data'].get('preprocess_module')
    if preprocess_module_path:
        spec = importlib.util.spec_from_file_location('feature_blocks', preprocess_module_path)
        feature_blocks_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_blocks_mod)
        feature_engineer = feature_blocks_mod.default
        FEATURE_BLOCKS = feature_blocks_mod.FEATURE_BLOCKS
        # 检查最优特征组合
        model_type = config['model']['type']
        feature_blocks_to_use = check_and_get_feature_blocks(model_type, FEATURE_BLOCKS)
        # 包装特征工程函数，确保只用指定组合
        feature_engineer_func = lambda df, train_df=None: feature_engineer(df, train_df, feature_blocks=feature_blocks_to_use)
    else:
        feature_engineer_func = None

    # 动态加载数据清洗模块
    clean_module_path = config['data'].get('clean_module')
    if clean_module_path:
        spec_clean = importlib.util.spec_from_file_location('clean_data', clean_module_path)
        clean_data_mod = importlib.util.module_from_spec(spec_clean)
        spec_clean.loader.exec_module(clean_data_mod)
        clean_data = clean_data_mod.default
    else:
        raise ValueError("请在config.yaml中配置data.clean_module")

    # 确保输出目录存在
    ensure_directories(config)
    
    # 1. 自动创建输出目录
    for d in [config['output']['models_dir'], config['output']['predictions_dir'], config['output']['images_dir']]:
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # 2. 数据加载
    df_train = pd.read_csv(config['data']['train_path'], sep=' ')
    df_test = pd.read_csv(config['data']['test_path'], sep=' ')

    # 3. 数据预处理
    df_train = preprocess_data(df_train, clean_data=clean_data, feature_engineer=feature_engineer_func)
    df_test = preprocess_data(df_test, train_df=df_train, clean_data=clean_data, feature_engineer=feature_engineer_func)



    prediction_column = config.get('prediction_column', 'price')
    id_column = config.get('id_column', 'SaleID')
    feature_cols = [col for col in df_train.columns if col not in [id_column, prediction_column]]
    X = preprocess_features(df_train, feature_cols)
    y = df_train[prediction_column]
    X_test = preprocess_features(df_test, feature_cols)
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    # 给 X 加上 config_output_images_dir 属性，供交叉验证流程使用
    X.config_output_images_dir = config['output']['images_dir']

    # 4. 训练配置
    train_config = RegressionTrainConfig(
        data=DataConfig(
            X=X,
            y=y,
            X_test=X_test,
            sale_ids=df_test[id_column],
            feature_cols=feature_cols,
            out_csv=os.path.join(config['output']['predictions_dir'], 'test_pred.csv')
        ),
        model=ModelConfig(
            model_type=config['model']['type'],
            save_model_path=None,
            save_model=False
        ),
        train=TrainConfig(
            feat_imp_path=os.path.join(config['output']['images_dir'], 'feature_importance.png'),
            log1p_target=False
        ),
        param_tuning=ParamTuningConfig(
            param_grid=config['model']['params'],
            use_cache=True,
            search_type='grid',
            n_iter=10,
            data_path=config['data']['train_path'],
            cache_path=config.get('cache', {}).get('param_cache_path', './cache/param_cache.json')
        )
    )

    # 5. 分段建模 or 主流回归
    if config['segmentation']['enabled']:
        segment_col = config['segmentation']['segment_col']
        n_bins = config['segmentation']['n_bins']
        # 只用auto_bins返回bins
        _, bins = auto_bins(df_train[segment_col], n_bins=n_bins)
        labels = list(range(n_bins))
        seg_config = SegmentedRegressionConfig(
            params=train_config,
            segment_col=segment_col,
            bins=bins,
            labels=labels,
            use_cv=config['cross_validation']['enabled'],
            n_splits=config['cross_validation']['n_splits'],
            random_state=42,
            verbose=True,
            df_train=df_train,
            df_test=df_test,
            mode=config['segmentation'].get('mode', 'segment')
        )
        maes_dict, final_model, segment_rule = segmented_regression(seg_config)
        print('分段各段MAE:', maes_dict)
        # 保存分段模型
        save_model(final_model, config['scripts']['segment_model_path'])
        print(f'分段模型已保存到 {config["scripts"]["segment_model_path"]}')
        # 保存分段特征重要性图（如支持）
        feat_imp_path = os.path.join(config['output']['images_dir'], 'feature_importance_segmented.png')
        if hasattr(final_model, 'feature_importances_'):
            plot_feature_importance(final_model, X, 20, feat_imp_path, config['model']['type'])
        else:
            print('当前模型不支持特征重要性图的生成。')
    else:
        # 主流回归/交叉验证流程
        if config['cross_validation']['enabled']:
            cv_config = CrossValidateConfig(
                params=train_config,
                n_splits=config['cross_validation']['n_splits'],
                random_state=42,
                verbose=True,
                use_cv=True
            )
            maes, mean_mae, final_model = cross_validate_regression(cv_config)
            print(f'交叉验证MAE: {mean_mae:.2f}')
            # 保存特征重要性图（如支持）
            feat_imp_path = os.path.join(config['output']['images_dir'], 'feature_importance.png')
            if hasattr(final_model, 'feature_importances_'):
                plot_feature_importance(final_model, X, 20, feat_imp_path, config['model']['type'])
            else:
                print('当前模型不支持特征重要性图的生成。')
        else:
            from src.models.regression.regression_train_predict import regression_train_predict
            final_model, mae = regression_train_predict(train_config)
            print(f'全量训练MAE: {mae:.2f}')
            # 保存特征重要性图（如支持）
            feat_imp_path = os.path.join(config['output']['images_dir'], 'feature_importance.png')
            if hasattr(final_model, 'feature_importances_'):
                plot_feature_importance(final_model, X, 20, feat_imp_path, config['model']['type'])
            else:
                print('当前模型不支持特征重要性图的生成。')

    # 6. 保存模型（主流回归）
    if not config['segmentation']['enabled']:
        model_save_path = os.path.join(config['output']['models_dir'], 'final_model.pkl')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        save_model(final_model, model_save_path)
        print(f'模型已保存到 {model_save_path}')

    # 7. 预测与结果保存
    if hasattr(final_model, 'predict'):
        preds_test = final_model.predict(X_test)
        test_pred_df = pd.DataFrame({
            id_column: df_test[id_column],
            prediction_column: preds_test
        })
        test_pred_path = os.path.join(config['output']['predictions_dir'], 'test_pred.csv')
        os.makedirs(os.path.dirname(test_pred_path), exist_ok=True)
        test_pred_df.to_csv(test_pred_path, index=False)
        print(f'测试集预测结果已保存到 {test_pred_path}')
        # 训练集预测与MAE
        preds_train = final_model.predict(X)
        train_pred_df = pd.DataFrame({
            id_column: df_train[id_column],
            prediction_column: preds_train
        })
        train_pred_path = os.path.join(config['output']['predictions_dir'], 'train_pred.csv')
        os.makedirs(os.path.dirname(train_pred_path), exist_ok=True)
        train_pred_df.to_csv(train_pred_path, index=False)
        print(f'训练集预测结果已保存到 {train_pred_path}')
    else:
        print('模型不支持predict方法，无法保存预测结果。') 