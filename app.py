from flask import Flask, render_template, request
import pandas as pd
# import numpy as np
import pickle
import shap
import base64
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__)

# 加载模型
with open('ann_model.pkl', 'rb') as f:
    model = pickle.load(f)

# # 添加调试信息
# print("Model type:", type(model))

# # 加载标准化工具
# with open('scaler_numeric.pkl', 'rb') as f1:
#     scaler = pickle.load(f1)

# 加载训练数据
X_train = pd.read_csv('X_train.csv')

# 确保X_train的列顺序与模型训练时一致
feature_order = ['年龄', '身高', '娱乐项目数', '化疗天数', 
                 '预期性二分类_1', '急性二分类_1', '性别_1',
                 '恶心呕吐控制效果_1', '看书绘画_1', '对恶心呕吐的预期_1',
                 '化疗方案分类_5', '化疗方案分类_6']
X_train = X_train[feature_order]

print("X_train shape:", X_train.shape)
print("X_train columns:", X_train.columns.tolist())

# 添加验证函数
def validate_input(form_data):
    errors = []
    
    # 年龄验证 (假设范围为0-18岁，因为是儿童医院)
    try:
        age = float(form_data['年龄'])
        if not 0 <= age <= 18:
            errors.append("年龄必须在0-18岁之间")
    except ValueError:
        errors.append("年龄必须是数字")

    # 身高验证 (假设范围为45-180cm)
    try:
        height = float(form_data['身高'])
        if not 45 <= height <= 180:
            errors.append("身高必须在45-180厘米之间")
    except ValueError:
        errors.append("身高必须是数字")

    # 娱乐项目数验证 (假设范围为0-10)
    try:
        activities = float(form_data['娱乐项目数'])
        if not 0 <= activities <= 10:
            errors.append("娱乐项目数必须在0-10之间")
    except ValueError:
        errors.append("娱乐项目数必须是数字")

    # 化疗天数验证 (假设范围为1-30)
    try:
        chemo_days = float(form_data['化疗天数'])
        if not 1 <= chemo_days <= 30:
            errors.append("化疗天数必须在1-30天之间")
    except ValueError:
        errors.append("化疗天数必须是数字")

    return errors

# 添加特征名称映射
feature_name_mapping = {
    '年龄': 'Age',
    '身高': 'Height',
    '娱乐项目数': 'Numbers of recreational activities',
    '化疗天数': 'Days of chemotherapy',
    '预期性二分类_1': 'Anticipatory CINV',
    '急性二分类_1': 'Acute CINV',
    '性别_1': 'Gender',
    '恶心呕吐控制效果_1': 'Control Effectiveness of CINV',
    '看书绘画_1': 'Reading and painting',
    '对恶心呕吐的预期_1': 'Expectancy of CINV',
    '化疗方案分类_5': 'Cisplatin-based regimens',
    '化疗方案分类_6': 'Other regimens'
}

def generate_shap_plot(input_features, model, X_train):
    # 创建带有英文特征名的DataFrame
    input_df = pd.DataFrame(
        input_features, 
        columns=feature_order
    ).rename(columns=feature_name_mapping)
    
    # 创建SHAP解释器
    explainer = shap.KernelExplainer(
        lambda x: model.predict_proba(x)[:, 1],
        shap.sample(X_train, 100)
    )
    
    # 计算SHAP值
    shap_values = explainer.shap_values(input_features)
    
    # 生成力图
    plt.figure(figsize=(12, 4))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_df.iloc[0],  # 使用带有英文名称的DataFrame
        matplotlib=True,
        show=False,
        text_rotation=45,  # 调整文本角度
        contribution_threshold=0.05
    )
    
    plt.tight_layout()  # 自动调整布局
    
    # 保存图片到内存
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close()
    
    # 转换为base64字符串
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 验证输入
        errors = validate_input(request.form)
        if errors:
            return render_template('error.html', errors=errors)
            
        # 1. 首先打印 scaler 的参数
        with open('scaler_numeric11.pkl', 'rb') as f:
            scaler_numeric = pickle.load(f)
        print("Scaler mean_:", scaler_numeric.mean_)
        print("Scaler scale_:", scaler_numeric.scale_)
        
        # 2. 创建数据框并打印原始值
        data = pd.DataFrame({
            '年龄': [float(request.form['年龄'])],
            '身高': [float(request.form['身高'])],
            '娱乐项目数': [float(request.form['娱乐项目数'])],
            '化疗天数': [float(request.form['化疗天数'])],
            '预期性二分类_1': [int(request.form['预期性二分类'])],
            '急性二分类_1': [int(request.form['急性二分类'])],
            '性别_1': [int(request.form['性别'])],
            '恶心呕吐控制效果_1': [int(request.form['恶心呕吐的控制效果'])],
            '看书绘画_1': [int(request.form['看书绘画'])],
            '对恶心呕吐的预期_1': [int(request.form['对恶心呕吐的预期'])],
            '化疗方案分类_5': [1 if request.form['化疗方案'] == '5' else 0],
            '化疗方案分类_6': [1 if request.form['化疗方案'] == '6' else 0]
        })
        print("Original data types:", data.dtypes)
        print("Original data:", data)

        # 3. 标准化连续变量
        continuous_vars = ['年龄', '身高', '娱乐项目数', '化疗天数']
        continuous_data = data[continuous_vars].values
        print("Before scaling:", continuous_data)
        scaled_continuous = scaler_numeric.transform(continuous_data)
        print("After scaling:", scaled_continuous)
        
        # 4. 更新数据框中的标准化值
        for i, col in enumerate(continuous_vars):
            data[col] = scaled_continuous[:, i]
        
        # 5. 确保特征顺序与训练时一致
        feature_order = ['年龄', '身高', '娱乐项目数', '化疗天数', 
                        '预期性二分类_1', '急性二分类_1', '性别_1',
                        '恶心呕吐控制效果_1', '看书绘画_1', '对恶心呕吐的预期_1',
                        '化疗方案分类_5', '化疗方案分类_6']
        
        input_features = data[feature_order].values
        print("Final input shape:", input_features.shape)
        print("Final input features:", input_features)

        # 6. 打印模型信息
        print("Model classes:", model.classes_)
        print("Model coefs shapes:", [coef.shape for coef in model.coefs_])
        
        # 进行预测
        probabilities = model.predict_proba(input_features)
        prediction = model.predict(input_features)
        
        # 返回结果
        risk_class = "高风险" if prediction[0] == 1 else "低风险"
        risk_probability = probabilities[0][1] if prediction[0] == 1 else probabilities[0][0]
        risk_percentage = round(risk_probability * 100, 2)


        # 生成SHAP力图
        shap_plot = generate_shap_plot(input_features, model, X_train)
        
        # 返回预测结果和力图
        return render_template(
            'result.html',
            risk_class=risk_class,
            risk_probability=risk_percentage,
            shap_plot=shap_plot
        )

    except Exception as e:
        print("Error occurred:", str(e))
        print("Error type:", type(e))
        import traceback
        print("Traceback:", traceback.format_exc())
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

