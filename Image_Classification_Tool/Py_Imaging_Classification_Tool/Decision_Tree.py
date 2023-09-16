import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import tree
from tqdm import tqdm

folder_path = 'C:\\CYLab\\data\\GFP_VO\\'
path = 'C:\\CYLab\\data\\GFP_VO\\0015_A6_0001.tif'
image = cv2.imread(path)

def calculate_network_stats(folder_path):
    num_images = 0
    num_bad_networks = 0
    num_good_networks = 0
    total_contours = 0

    list_abundant = []
    list_scarcity = []

    with tqdm(total=len(os.listdir(folder_path)), desc='Processing') as pbar:
        for filename in os.listdir(folder_path):
            if filename.endswith('.tif'):
                num_images += 1
                path = os.path.join(folder_path, filename)

                # 进行图像处理
                image = cv2.imread(path)
                # 影像預處理
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 轉成灰階
                _, binary_image = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY)  # 二值化
                blur_image = cv2.medianBlur(binary_image, 3)  # 中值濾波 減輕噪聲
                blur_image_twice = cv2.medianBlur(blur_image, 3)  # 平滑圖像

                # 進行形態學處理來去除噪點和小的斷點，可以使用開運算操作
                kernel = np.ones((3, 3), np.uint8)
                opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
                # 使用Canny邊緣檢測算法來檢測影像中的邊緣
                canny = cv2.Canny(opening, 50, 150)
                # 尋找影像中的輪廓，可以使用cv2.findContours函數，設定輪廓檢測模式為cv2.RETR_TREE，輪廓近似方法為cv2.CHAIN_APPROX_SIMPLE
                contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # 計算網絡分布的度量，可以計算影像中輪廓的數量、面積、周長等指標，這些指標可以用來評估網絡的分布情況
                num_contours = len(contours)
                total_area = sum([cv2.contourArea(cnt) for cnt in contours])
                total_length = sum([cv2.arcLength(cnt, True) for cnt in contours])

                # Flag judgement
                flag = 1

                if num_contours < 1000 and total_area < 25000 and total_length < 35000:
                    flag = 0
                    num_bad_networks += 1
                    list_scarcity.append([filename, flag])
                else:
                    flag = 1
                    num_good_networks += 1
                    list_abundant.append([filename, flag])

                if num_images % 80 == 0 or num_images == len(os.listdir(folder_path)):
                    print(f"Processed {num_images} images.")

                # 在進度條中更新進度
                pbar.update(1)

                total_contours += num_contours

# calculate_network_stats(folder_path)


# import os
# import cv2
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# def extract_features(image_path):
#     path = os.path.join(folder_path, image_path)
#
#     # 进行图像处理
#     image = cv2.imread(path)
#     # 影像預處理
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 轉成灰階
#     _, binary_image = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY)  # 二值化
#     blur_image = cv2.medianBlur(binary_image, 3)  # 中值濾波 減輕噪聲
#     blur_image_twice = cv2.medianBlur(blur_image, 3)  # 平滑圖像
#
#     # 進行形態學處理來去除噪點和小的斷點，可以使用開運算操作
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
#     # 使用Canny邊緣檢測算法來檢測影像中的邊緣
#     canny = cv2.Canny(opening, 50, 150)
#     # 尋找影像中的輪廓，可以使用cv2.findContours函數，設定輪廓檢測模式為cv2.RETR_TREE，輪廓近似方法為cv2.CHAIN_APPROX_SIMPLE
#     contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # 計算網絡分布的度量，可以計算影像中輪廓的數量、面積、周長等指標，這些指標可以用來評估網絡的分布情況
#     num_contours = len(contours)
#     total_area = sum([cv2.contourArea(cnt) for cnt in contours])
#     total_length = sum([cv2.arcLength(cnt, True) for cnt in contours])
#
#     # 返回特征向量
#     return [num_contours, total_area, total_length]
#
# def classify_networks(folder_path):
#     # 存储特征和标签
#     num_images=0
#     features = []
#     labels = []
#     with tqdm(total=len(os.listdir(folder_path)), desc='Processing') as pbar:
#         for filename in os.listdir(folder_path):
#             if filename.endswith('.tif'):
#                 num_images += 1
#                 path = os.path.join(folder_path, filename)
#
#                 # 提取特征
#                 features.append(extract_features(path))
#
#                 num_contours, total_area, total_length = extract_features(path)
#
#                 # 判断网络分布是否好，标记标签
#                 if num_contours < 1000 and total_area < 25000 and total_length < 35000:
#                     labels.append(0)
#                 else:
#                     labels.append(1)
#
#                 if num_images % 80 == 0 or num_images == len(os.listdir(folder_path)):
#                     print(f"Processed {num_images} images.")
#
#                 # 在進度條中更新進度
#                 pbar.update(1)
#
#     # 将特征和标签转换为NumPy数组
#     features = np.array(features)
#     labels = np.array(labels)
#
#     # 划分训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#
#     # 创建决策树模型并训练
#     model = DecisionTreeClassifier()
#     model.fit(X_train, y_train)
#
#     # 在测试集上进行预测
#     y_pred = model.predict(X_test)
#
#     # 计算准确率
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy:", accuracy)
#
# # 指定包含多个影像的文件夹路径
# classify_networks(folder_path)
import os
import cv2
import numpy as np
import sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import tempfile


def extract_features(image_path):
    path = os.path.join(folder_path, image_path)

    # 进行图像处理
    image = cv2.imread(path)
    # 影像預處理
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 轉成灰階
    _, binary_image = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY)  # 二值化
    blur_image = cv2.medianBlur(binary_image, 3)  # 中值濾波 減輕噪聲
    blur_image_twice = cv2.medianBlur(blur_image, 3)  # 平滑圖像

    # 進行形態學處理來去除噪點和小的斷點，可以使用開運算操作
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    # 使用Canny邊緣檢測算法來檢測影像中的邊緣
    canny = cv2.Canny(opening, 50, 150)
    # 尋找影像中的輪廓，可以使用cv2.findContours函數，設定輪廓檢測模式為cv2.RETR_TREE，輪廓近似方法為cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 計算網絡分布的度量，可以計算影像中輪廓的數量、面積、周長等指標，這些指標可以用來評估網絡的分布情況
    num_contours = len(contours)
    total_area = sum([cv2.contourArea(cnt) for cnt in contours])
    total_length = sum([cv2.arcLength(cnt, True) for cnt in contours])

    # 返回特征向量
    return [num_contours, total_area, total_length]
    # return total_area, total_length

# def augment_samples(features, labels, num_augmentations):
#     augmented_features = []
#     augmented_labels = []
#     global index
#     index = 0
#     for _ in range(len(labels)):
#         if labels[_]==1 :
#             index +=1
#     with tqdm(total=index*num_augmentations, desc='Processing') as pbar:
#         for i in range(len(features)):
#             # 增强当前样本
#             augmented_features.append(features[i])
#             augmented_labels.append(labels[i])
#             if labels[i] == 1:
#                 for _ in range(num_augmentations):
#                     # 生成增强样本
#                     augmented_image = cv2.flip(image, 1)  # 使用图像处理技术生成增强图像
#                     # 创建临时文件来保存增强图像
#                     temp_path = tempfile.mktemp(suffix='.tif')
#                     cv2.imwrite(temp_path, augmented_image)
#
#                     # 提取增强图像的特征
#                     augmented_feature = extract_features(temp_path)
#
#                     # 删除临时文件
#                     os.remove(temp_path)
#                     augmented_features.append(augmented_feature)
#                     augmented_labels.append(labels[i])
#                     pbar.update(1)
#                 else:
#                     pbar.update(1)
#
#         augmented_features = np.array(augmented_features)
#         augmented_labels = np.array(augmented_labels)
#
#
#     return augmented_features, augmented_labels
def classify_networks(folder_path):
    # 存储特征和标签
    num_images = 0
    features = []
    labels = []
    with tqdm(total=len(os.listdir(folder_path)), desc='Processing') as pbar:
        for filename in os.listdir(folder_path):
            if filename.endswith('.tif'):
                num_images += 1
                path = os.path.join(folder_path, filename)

                # 提取特征
                features.append(extract_features(path))

                num_contours, total_area, total_length = extract_features(path)
                # total_area, total_length = extract_features(path)

                # 判断网络分布是否好，标记标签
                # if num_contours < 1000 and total_area < 25000 and total_length < 35000:
                if total_area < 25000 and total_length < 35000:
                    labels.append(0)
                else:
                    labels.append(1)

                if num_images % 80 == 0 or num_images == len(os.listdir(folder_path)):
                    print(f"Processed {num_images} images.")

                # 在进度条中更新进度
                pbar.update(1)
    feature_names = ['Number of Contours', 'Total Area', 'Total Length']
    # feature_names = ['Total Area', 'Total Length']

    # 将特征和标签转换为NumPy数组
    features = np.array(features)
    labels = np.array(labels)

    #特徵增強

    # # 样本扩充 ##增加數量沒意義, 因為是做分類, 而不是學習影像
    # num_augmentations=2
    # augmented_features, augmented_labels = augment_samples(features, labels, num_augmentations)

    # # 特征正则化
    # scaler = StandardScaler()
    # augmented_features = scaler.fit_transform(augmented_features)

    # # 数据集拆分
    # X_train, X_test, y_train, y_test = train_test_split(augmented_features, augmented_labels, test_size=0.2,
    #                                                     random_state=42)

    # 特征正则化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    # print(features)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # 使用交叉验证寻找最佳模型参数
    param_grid = {'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10]}
    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # 创建决策树模型并训练
    model = DecisionTreeClassifier(max_depth=best_params['max_depth'],
                                   min_samples_split=best_params['min_samples_split'])
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # 使用交叉验证评估模型性能
    cv_scores = cross_val_score(model, features, labels, cv=5)
    print("Cross Validation Scores:", cv_scores)
    print("Mean CV Score:", np.mean(cv_scores))

    # 绘制决策树
    plt.figure(figsize=(12, 6))
    tree.plot_tree(model, feature_names=["num_contours", "total_area", "total_length"], class_names=["Bad Network", "Good Network"], filled=True)
    plt.show()

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 计算精确度
    precision = precision_score(y_test, y_pred) #预测为正例的样本中的真实正例的识别能力

    # 计算召回率
    recall = recall_score(y_test, y_pred) #实际为正例的样本中，模型预测为正例的比例

    # 计算 F1 值
    f1 = f1_score(y_test, y_pred)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    # 可视化混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # 在每个单元格中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.show()
    # 计算 ROC 曲线和 AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    #绘制 ROC 曲线
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

    # 计算特征重要性
    feature_importance = model.feature_importances_
    print("Feature Importance:")
    for i, importance in enumerate(feature_importance):
        # print(f"Feature {i + 1}: {importance}")
        feature_name = feature_names[i]
        print(f"{feature_name}: {importance}")

    # 模型集成
    # ensemble_model = RandomForestClassifier(n_estimators=10)
    # ensemble_model.fit(X_train, y_train)
    # y_ensemble_pred = ensemble_model.predict(X_test)
    # ensemble_accuracy = accuracy_score(y_test, y_ensemble_pred)
    # print("Ensemble Accuracy:", ensemble_accuracy)

# 指定包含多个影像的文件夹路径
classify_networks(folder_path)
