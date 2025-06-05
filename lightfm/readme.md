Es wurde zuerst 3 LightFM Modelle erstellt und vergliechen: WARP, BPR und WARP-Kos.
Es wurden dann Hyperparameter-Tuning (Optuna) und Feature Engineering (Item-Metadaten)
in das beste Modell angewendet.

### 🔍 Finale Ergebnisse (bestes Modell: WARP)
Trotz Optimierungen blieben die Top-N-Metriken moderat:

- Precision@10: **0.1055**
- Recall@10: **0.0727**
- NDCG@10: **0.1091**
- AUC: **0.9582**

---

### 📊 Vergleich: WARP vs BPR vs WARP-KOS

**🔹 WARP Modell:**
- Train Precision@10: 0.5397  
- Test Precision@10:  0.1180  
- Train Recall@10:    0.0637  
- Test Recall@10:     0.0552  
- Train NDCG@10:      0.5655  
- Test NDCG@10:       0.1204  
- Train AUC:          0.9694  
- Test AUC:           0.9529  

**🔹 BPR Modell:**
- Train Precision@10: 0.4895  
- Test Precision@10:  0.0890  
- Train Recall@10:    0.0572  
- Test Recall@10:     0.0418  
- Train NDCG@10:      0.5428  
- Test NDCG@10:       0.0942  
- Train AUC:          0.9246  
- Test AUC:           0.9022  

**🔹 WARP-KOS Modell:**
- Train Precision@10: 0.5406  
- Test Precision@10:  0.1142  
- Train Recall@10:    0.0638  
- Test Recall@10:     0.0532  
- Train NDCG@10:      0.5457  
- Test NDCG@10:       0.1097  
- Train AUC:          0.9221  
- Test AUC:           0.9115  

---

Alle Modelle erreichen eine hohe AUC, aber Precision, Recall und NDCG bleiben durch die hohe Daten-Sparsity begrenzt. Feature Engineering und Tuning helfen, aber der Cold-Start- und Top-N-Empfehlung bleibt herausfordernd.
