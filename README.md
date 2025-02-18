# Pairon
本專案為個人碩論[整合深度學習與多重特徵提取之人物辨識及全身追蹤系統](https://hdl.handle.net/11296/8ytxzm)的後端實作部分，基於視覺基礎模型(vision foundation model)進行零樣本(Zero-shot)的人物辨識與追蹤系統。

## 專案概述
Pairon 是一套整合深度學習與多重特徵提取的人物辨識及全身追蹤系統。 本系統突破傳統人臉辨識的限制，能夠對有遮擋或不完整的人臉擷取特徵，並透過零樣本學習實現各種情況下的人臉特徵辨識，進而將辨識到的人物進行持續追蹤。

## 專案流程
```mermaid
sequenceDiagram
    participant SysAdmin as 系統管理模組
    participant ModelLoader as 模型載入模組
    participant CacheManager as 快取管理模組
    participant FeatureProc as 特徵處理模組
    participant Storage as 儲存模組
    participant ResourceMon as 資源監控模組

    %% Model Initialization Phase
    Note over SysAdmin,ResourceMon: 模型初始化階段
    SysAdmin->>ModelLoader: 1. 初始化模型請求
    ModelLoader->>CacheManager: 2. 檢查模型快取
    CacheManager-->>ModelLoader: 3. 回傳快取狀態
    
    alt 快取存在且有效
        ModelLoader-->>SysAdmin: 4a. 載入快取參數
    else 需要重新載入
        ModelLoader-->>SysAdmin: 4b. 載入模型參數
    end
    
    SysAdmin->>ModelLoader: 5. 更新載入
    ModelLoader->>CacheManager: 6. 更新快取
    ModelLoader-->>SysAdmin: 7. 模型初始化完成

    %% Feature Processing Phase
    Note over SysAdmin,ResourceMon: 特徵處理與監控階段
    loop 特徵處理循環
        SysAdmin->>FeatureProc: 1. 開始特徵處理
        FeatureProc->>Storage: 2. 查詢特徵索引
        Storage-->>FeatureProc: 3. 回傳索引資訊
        
        FeatureProc->>Storage: 4. 特徵向量儲存
        Storage-->>FeatureProc: 5. 儲存確認
        FeatureProc->>Storage: 6. 完成特徵儲存
        Storage->>ResourceMon: 7. 執行一致性檢查
        Storage-->>FeatureProc: 8. 處理完成確認
        FeatureProc-->>SysAdmin: 9. 確認處理完成
    end
```

### 範例影片

[![系統展示影片](https://img.youtube.com/vi/gJidwmYNd6A/0.jpg)](https://www.youtube.com/watch?v=gJidwmYNd6A)
## 辨識效能

### 資料集
- 使用 **Market-1501** 與**Market-1501 Attribute** 資料集進行實驗，並結合其附帶的屬性標註進行人物屬性辨識評估，遮擋情境追蹤為利用自行設計的影像資料進行實驗。

### 人物辨識性能
- **Rank-1 準確率**：92.99%
- **Rank-5 準確率**：97.18%
- **Rank-10 準確率**：97.80%

### 人物屬性辨識
- 配件與服裝顏色辨識準確率均超過 95%

### 遮擋情境追蹤
- 系統成功應對九種不同類型的遮擋情境（例如動態人群遮擋、機車交會等），在追蹤穩定性上顯著超越傳統方法（如 ByteTrack 與 BoT-SORT）。

## 環境需求

本系統需要以下模型檔案才能正常運行：
- GFPGAN 模型檔：用於人臉修復與增強
  - 下載位置：[GFPGAN Model](https://drive.google.com/drive/folders/1AspP1c836z_abNLn1REQNvXQvnQ43zBR?usp=sharing)
  - 將下載的模型放入根目錄下
