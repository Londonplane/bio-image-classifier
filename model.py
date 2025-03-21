from PIL import Image
import numpy as np
import os
import requests
import json
import base64
from io import BytesIO
from dotenv import load_dotenv

class ImageClassifier:
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        self.api_key = os.getenv('CLARIFAI_API_KEY')
        if not self.api_key:
            raise ValueError("请设置CLARIFAI_API_KEY环境变量")
            
        # 定义我们的三个类别与通用类别的映射
        self.category_mapping = {
            'person': ['person', 'man', 'woman', 'boy', 'girl', 'child', 'baby', 'human',
                      'face', 'portrait', 'people', 'adult', 'teenager'],
            'animal': ['animal', 'bird', 'cat', 'dog', 'fish', 'insect', 'bear', 'lion', 'tiger', 
                      'elephant', 'monkey', 'horse', 'sheep', 'cow', 'rabbit', 'snake',
                      'turtle', 'whale', 'zebra', 'fox', 'wolf', 'frog', 'butterfly',
                      'bee', 'spider', 'snail', 'worm', 'crab', 'lobster', 'jellyfish',
                      'pet', 'wildlife', 'zoo'],
            'plant': ['plant', 'flower', 'tree', 'vegetable', 'fruit', 'grass', 'herb',
                     'mushroom', 'corn', 'cabbage', 'artichoke', 'bell_pepper', 'cardoon',
                     'broccoli', 'cauliflower', 'zucchini', 'acorn', 'hip', 'ear', 'rapeseed',
                     'garden', 'forest', 'jungle', 'woods', 'leaf', 'petal', 'stem']
        }
        
        # 备用分类方法（如果API调用失败）
        self.color_thresholds = {
            'person': {'r': 0.55, 'g': 0.45, 'b': 0.40},  # 人类肤色偏红
            'animal': {'r': 0.45, 'g': 0.40, 'b': 0.35},  # 动物颜色多样
            'plant': {'r': 0.35, 'g': 0.55, 'b': 0.30}    # 植物偏绿色
        }
    
    def classify_with_api(self, image_path):
        """使用Clarifai API进行图像分类"""
        try:
            # 读取图像并转换为base64
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Clarifai API URL
            api_url = "https://api.clarifai.com/v2/models/general-image-recognition/outputs"
            
            # 使用环境变量中的API密钥
            headers = {
                'Authorization': f'Key {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # 准备请求数据
            payload = {
                "inputs": [
                    {
                        "data": {
                            "image": {
                                "base64": encoded_string
                            }
                        }
                    }
                ]
            }
            
            # 发送请求
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                concepts = result['outputs'][0]['data']['concepts']
                
                # 将概念映射到我们的三个类别
                category_scores = {
                    'person': 0.0,
                    'animal': 0.0,
                    'plant': 0.0
                }
                
                # 计算每个类别的总分数
                for concept in concepts[:20]:  # 只考虑前20个概念
                    concept_name = concept['name'].lower()
                    confidence = concept['value']  # 已经是0-1范围
                    
                    # 检查该概念属于我们的哪个大类
                    for category, keywords in self.category_mapping.items():
                        for keyword in keywords:
                            if keyword in concept_name:
                                category_scores[category] += confidence
                                break
                
                # 如果没有匹配的类别，使用备用方法
                if sum(category_scores.values()) < 0.1:
                    return None
                
                # 找出得分最高的类别
                predicted_category = max(category_scores, key=category_scores.get)
                confidence = category_scores[predicted_category]
                
                return predicted_category, confidence
            
            return None
        except Exception as e:
            print(f"API调用失败: {e}")
            return None
    
    def extract_features(self, image):
        """提取图像特征（备用方法）"""
        # 调整图像大小以加快处理速度
        image = image.resize((200, 200))
        
        # 将图像转换为numpy数组
        img_array = np.array(image)
        
        # 计算平均RGB值
        avg_r = np.mean(img_array[:, :, 0]) / 255.0
        avg_g = np.mean(img_array[:, :, 1]) / 255.0
        avg_b = np.mean(img_array[:, :, 2]) / 255.0
        
        # 计算RGB标准差（颜色多样性）
        std_r = np.std(img_array[:, :, 0]) / 255.0
        std_g = np.std(img_array[:, :, 1]) / 255.0
        std_b = np.std(img_array[:, :, 2]) / 255.0
        
        # 计算颜色区域比例
        # 肤色区域（用于检测人）
        skin_mask = ((img_array[:,:,0] > 60) & (img_array[:,:,0] < 200) & 
                     (img_array[:,:,1] > 40) & (img_array[:,:,1] < 170) & 
                     (img_array[:,:,2] > 20) & (img_array[:,:,2] < 170))
        skin_ratio = np.sum(skin_mask) / (200 * 200)
        
        # 绿色区域（用于检测植物）
        green_mask = ((img_array[:,:,1] > img_array[:,:,0]) & 
                      (img_array[:,:,1] > img_array[:,:,2]))
        green_ratio = np.sum(green_mask) / (200 * 200)
        
        # 纹理特征 - 简单的边缘检测
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        h_gradient = np.abs(gray[:, 1:] - gray[:, :-1])
        v_gradient = np.abs(gray[1:, :] - gray[:-1, :])
        avg_gradient = (np.mean(h_gradient) + np.mean(v_gradient)) / 2.0 / 255.0
        
        return {
            'avg_color': (avg_r, avg_g, avg_b),
            'std_color': (std_r, std_g, std_b),
            'skin_ratio': skin_ratio,
            'green_ratio': green_ratio,
            'avg_gradient': avg_gradient
        }
    
    def classify_with_features(self, features):
        """基于特征对图像进行分类（备用方法）"""
        avg_r, avg_g, avg_b = features['avg_color']
        std_r, std_g, std_b = features['std_color']
        skin_ratio = features['skin_ratio']
        green_ratio = features['green_ratio']
        avg_gradient = features['avg_gradient']
        
        scores = {
            'person': 0,
            'animal': 0,
            'plant': 0
        }
        
        # 基于颜色距离的评分
        for category, thresholds in self.color_thresholds.items():
            r_dist = (avg_r - thresholds['r']) ** 2
            g_dist = (avg_g - thresholds['g']) ** 2
            b_dist = (avg_b - thresholds['b']) ** 2
            color_dist = np.sqrt(r_dist + g_dist + b_dist)
            scores[category] += (1.0 - color_dist) * 0.3  # 颜色距离权重为0.3
        
        # 基于肤色比例的评分
        scores['person'] += skin_ratio * 0.4  # 肤色比例权重为0.4
        
        # 基于绿色比例的评分
        scores['plant'] += green_ratio * 0.4  # 绿色比例权重为0.4
        
        # 基于颜色多样性的评分
        color_diversity = (std_r + std_g + std_b) / 3.0
        scores['animal'] += color_diversity * 0.3  # 动物颜色多样性权重为0.3
        
        # 基于纹理的评分
        scores['animal'] += avg_gradient * 0.2  # 动物纹理通常更复杂
        scores['person'] -= avg_gradient * 0.1  # 人类肤色通常较为平滑
        
        # 额外的启发式规则
        # 如果图像非常绿，很可能是植物
        if green_ratio > 0.5:
            scores['plant'] += 0.2
        
        # 如果图像有大量肤色，很可能是人
        if skin_ratio > 0.3:
            scores['person'] += 0.2
        
        # 如果图像颜色多样且纹理复杂，很可能是动物
        if color_diversity > 0.2 and avg_gradient > 0.1:
            scores['animal'] += 0.2
        
        return scores
    
    def predict(self, image_path):
        """预测图像类别"""
        # 首先尝试使用API进行分类
        api_result = self.classify_with_api(image_path)
        
        if api_result:
            predicted_category, confidence = api_result
        else:
            # 如果API调用失败，使用备用方法
            print("API调用失败，使用备用方法...")
            # 打开图像
            image = Image.open(image_path).convert('RGB')
            
            # 提取特征
            features = self.extract_features(image)
            
            # 分类图像
            scores = self.classify_with_features(features)
            
            # 找出得分最高的类别
            predicted_category = max(scores, key=scores.get)
            confidence = scores[predicted_category]
            
            # 归一化置信度到0-1范围
            total_score = sum(scores.values())
            if total_score > 0:
                confidence = confidence / total_score
        
        # 将英文类别转换为中文
        category_translation = {
            'person': '人',
            'animal': '动物',
            'plant': '植物'
        }
        
        return category_translation[predicted_category], confidence 