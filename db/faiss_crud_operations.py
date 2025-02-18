import faiss
import numpy as np
from typing import List, Tuple, Union, Dict
import os
import time

class FaissManager:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # 使用 L2 距離
        self.data = []  # 用於儲存向量和附加資料

    def insert(self, name: str, embedding: Union[List[float], np.ndarray], subset_id: int = 0, timestamp: float = None, features: List[str] = None) -> None:
        """插入一個向量和對應的名稱、子集ID、時間戳和特徵"""
        if isinstance(embedding, list):
            embedding = np.array([embedding])  # 轉換成 2D array
        elif isinstance(embedding, np.ndarray):
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
        
        if embedding.shape[1] != self.dimension:
            raise ValueError(f"向量維度不匹配。預期 {self.dimension}，實際得到 {embedding.shape[1]}")
        print('faiss embedding:', len(embedding))    
        embedding = embedding.astype('float32')
        self.index.add(embedding)
        if timestamp is None:
            timestamp = time.time()
        self.data.append({
            'name': name,
            'subset_id': subset_id,
            'embedding': embedding,
            'timestamp': timestamp,
            'features': features  # 新增特徵
        })

    def save_index(self, file_path: str) -> None:
        """保存索引到文件"""
        try:
            faiss.write_index(self.index, file_path)
        except Exception as e:
            raise IOError(f"保存索引時發生錯誤: {str(e)}")

    def load_index(self, file_path: str) -> None:
        """從文件加載索引"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"索引文件不存在: {file_path}")
        try:
            self.index = faiss.read_index(file_path)
        except Exception as e:
            raise IOError(f"加載索引時發生錯誤: {str(e)}")

    def search(self, query_embedding: Union[List[float], np.ndarray], top_k: int = 10) -> List[Dict[str, Union[str, float]]]:
        """搜索最相似的向量並返回附加資料"""
        if isinstance(query_embedding, list):
            query_embedding = np.array([query_embedding])
        elif isinstance(query_embedding, np.ndarray):
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
                
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"查詢向量維度不匹配。預期 {self.dimension}，實際得到 {query_embedding.shape[1]}")
            
        query_embedding = query_embedding.astype('float32')
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.data)))
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # faiss 可能返回 -1 表示找不到足夠的結果
                data = self.data[idx]
                results.append({
                    'name': data['name'],
                    'subset_id': data['subset_id'],
                    'distance': float(dist),
                    'timestamp': data['timestamp'],
                    'features': data.get('features')  # 返回特徵
                })
        return results
    
    def search_within_timestamp_range(self, query_embedding: Union[List[float], np.ndarray], start_timestamp: float, end_timestamp: float, top_k: int = 10) -> List[Dict[str, Union[str, float]]]:
        """搜索特定時間戳範圍內的最相似向量並返回附加資料"""
        if isinstance(query_embedding, list):
            query_embedding = np.array([query_embedding])
        elif isinstance(query_embedding, np.ndarray):
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
                
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"查詢向量維度不匹配。預期 {self.dimension}，實際得到 {query_embedding.shape[1]}")
            
        query_embedding = query_embedding.astype('float32')
        
        # 過濾出特定時間戳範圍內的向量
        filtered_data = [data for data in self.data if start_timestamp <= data['timestamp'] <= end_timestamp]
        if not filtered_data:
            return []
        
        filtered_embeddings = np.array([data['embedding'] for data in filtered_data])
        filtered_embeddings = filtered_embeddings.reshape(len(filtered_data), self.dimension)
        
        # 創建一個臨時索引來搜索過濾後的向量
        temp_index = faiss.IndexFlatL2(self.dimension)
        temp_index.add(filtered_embeddings)
        
        distances, indices = temp_index.search(query_embedding, min(top_k, len(filtered_data)))
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # faiss 可能返回 -1 表示找不到足夠的結果
                data = filtered_data[idx]
                results.append({
                    'name': data['name'],
                    'subset_id': data['subset_id'],
                    'distance': float(dist),
                    'timestamp': data['timestamp'],
                    'features': data.get('features')  # 返回特徵
                })
        return results
    
    def search_within_last_n_seconds(self, query_embedding: Union[List[float], np.ndarray], n_frames: float, top_k: int = 10) -> List[Dict[str, Union[str, float]]]:
        """搜索最新embedding的前 n 秒內的最相似向量並返回附加資料"""
        if isinstance(query_embedding, list):
            query_embedding = np.array([query_embedding])
        elif isinstance(query_embedding, np.ndarray):
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
                
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"查詢向量維度不匹配。預期 {self.dimension}，實際得到 {query_embedding.shape[1]}")
            
        query_embedding = query_embedding.astype('float32')
        
        # 獲取最新的時間戳
        latest_timestamp = max(data['timestamp'] for data in self.data)
        start_timestamp = latest_timestamp - n_frames
        
        # 過濾出最近 n 秒內的向量
        filtered_data = [data for data in self.data if start_timestamp <= data['timestamp'] <= latest_timestamp]
        if not filtered_data:
            return []
        
        filtered_embeddings = np.array([data['embedding'] for data in filtered_data])
        filtered_embeddings = filtered_embeddings.reshape(len(filtered_data), self.dimension)
        
        # 創建一個臨時索引來搜索過濾後的向量
        temp_index = faiss.IndexFlatL2(self.dimension)
        temp_index.add(filtered_embeddings)
        
        distances, indices = temp_index.search(query_embedding, min(top_k, len(filtered_data)))
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # faiss 可能返回 -1 表示找不到足夠的結果
                data = filtered_data[idx]
                results.append({
                    'name': data['name'],
                    'subset_id': data['subset_id'],
                    'distance': float(dist),
                    'timestamp': data['timestamp'],
                    'features': data.get('features')  # 返回特徵
                })
        return results

class FaissCRUD:
    def __init__(self, dimension: int):
        self.manager = FaissManager(dimension)
        
    def insert_body_embedding(self, name: str, embedding: Union[List[float], np.ndarray], subset_id: int = 0, file_path: str = "faiss_index.bin", timestamp: float = None, features: List[str] = None) -> None:
        """插入向量並保存索引"""
        try:
            self.manager.insert(name, embedding, subset_id, timestamp, features)
            self.manager.save_index(file_path)
        except Exception as e:
            raise Exception(f"插入向量時發生錯誤: {str(e)}")
        
    def save_index(self, file_path: str) -> None:
        self.manager.save_index(file_path)

    def load_index(self, file_path: str) -> None:
        self.manager.load_index(file_path)

    def search_similar_body(self, query_embedding: Union[List[float], np.ndarray], top_k: int = 10) -> List[Dict[str, Union[str, float]]]:
        """搜索相似向量並返回附加資料"""
        return self.manager.search(query_embedding, top_k)
    
    def search_within_timestamp_range(self, query_embedding: Union[List[float], np.ndarray], start_timestamp: float, end_timestamp: float, top_k: int = 10) -> List[Dict[str, Union[str, float]]]:
        """搜索特定時間戳範圍內的最相似向量並返回附加資料"""
        return self.manager.search_within_timestamp_range(query_embedding, start_timestamp, end_timestamp, top_k)
    
    def search_within_last_n_seconds(self, query_embedding: Union[List[float], np.ndarray], n_frames: float, top_k: int = 10) -> List[Dict[str, Union[str, float]]]:
        """搜索最近 n 秒內的最相似向量並返回附加資料"""
        return self.manager.search_within_last_n_seconds(query_embedding, n_frames, top_k)