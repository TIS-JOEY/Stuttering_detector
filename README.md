# Stuttering_detector
現今智能虛擬助理面臨的一大挑戰，即是使用者常常因各種原因而無法完整表達句子或太慢，尤其是銀髮族或身心障礙者等等，導致智能虛擬助理無法正確的識別使用者的意圖，而本專案目的即是要解決此問題。

# Usage

預設一個情境是使用者說話太慢，導致ASR辨識不完全，產生句子邊界偵測的錯誤。
輸入語句：我我我想要吃義大利麵
```
import BILSTM_CNN_CRF as stutter_detect

p_data = stutter_detect.processData()
p_data.loadParameters()

bilstm = stutter_detect.BI_LSTM_CNN_CRF(p_data.word2id,p_data.id2word,p_data.tag2id,p_data.id2tag,training_data = p_data.training_data)

bilstm.predict(list(jieba.cut('我我我想要')))
# --> [('我','O'),('我','O'),('我','O'),('想要','O')]，並未出現'COMMENT'或'PERIOD'的標記，代表句子未完全，應存至暫存器，待下句輸入語句一同標記。

bilstm.predict(list(jieba.cut('我我我想要吃義大利麵')))
# --> [('我','O'),('我','O'),('我','O'),('想要','O'),('吃','O'),('義大利麵','PERIOD')]，出現'PERIOD'，代表輸入句子為完整句子，可進行後續分析。
```

# Skill

本專案提供BILSTM-CRF與BILSTM-CNN-CRF兩種序列標籤標記模型，據實驗結果，BILSTM-CNN-CRF表現較好。
