# Model Mimarisi Akış Şeması

Aşağıda `BP_TripleHybrid` modelinin `forward` metodunun görsel bir akışı bulunmaktadır.

```mermaid
flowchart TD
    %% --- BÖLÜM 1: LR RANGE TEST (Değerler Doldurulmuş) ---
    subgraph LR Range Test ["Amaç: base_lr ve max_lr Bulma (Leslie Smith Yöntemi)"]
        direction TB
        LRT_A["Başla: lr_range_test"]
        
        LRT_B["<b>1. Adım Hesaplayıcıyı (mult) Ayarla:</b><br>num = len(train_loader) - 1<br>mult = (0.1 / 1e-8) ** (1/num)"]
        LRT_C["<b>2. Başlangıç Değerlerini Ayarla:</b><br>lr = 1e-8 (init_value)<br>Optimizer = Lookahead(Adam(lr))<br>beta = 0.98, avg_loss = 0, best_loss = inf"]
        
        LRT_D{"<b>3. Batch Döngüsü</b> (train_loader üzerinden)"}
        
        LRT_E["<b>4. Modeli Eğit:</b><br>optimizer.zero_grad()<br>loss = criterion(model(inputs), targets)<br>loss.backward()<br>optimizer.step() <i>(Lookahead adımı)</i>"]
        
        LRT_F["<b>5. Kaybı Yumuşat (beta=0.98):</b><br>avg_loss = 0.98 * avg_loss + 0.02 * loss<br>smoothed_loss = avg_loss / (1 - 0.98**(batch+1))"]
        
        LRT_G["<b>6. Değerleri Kaydet:</b><br>log_lrs.append(log10(lr))<br>losses.append(smoothed_loss)"]
        
        LRT_H{"if smoothed_loss < best_loss"}
        LRT_H -- Evet --> LRT_I["best_loss = smoothed_loss"]
        LRT_H -- Hayır --> LRT_J
        
        LRT_I --> LRT_J{"if smoothed_loss > 4 * best_loss<br>(Kayıp patladı mı?)"}
        
        LRT_J -- Hayır (Devam) --> LRT_K["<b>7. LR'yi Artır (Logaritmik Adım):</b><br>lr = lr * mult<br>optimizer.param_groups['lr'] = lr"]
        LRT_K --> LRT_D
        
        LRT_J -- Evet (Durdur) --> LRT_L["Döngüyü Durdur (break)"]
        
        LRT_L --> LRT_M["<b>8. Analiz Et:</b><br>Grafiği çiz (log_lrs vs losses)"]
        LRT_M --> LRT_N["<b>base_lr</b> = np.gradient(losses).argmin()<br>(En hızlı düşüş noktası)"]
        LRT_M --> LRT_O["<b>max_lr</b> = losses.argmin()<br>(En düşük kayıp noktası)"]
        
        LRT_N --> LRT_SON["Çıktı: (base_lr, max_lr)"]
        LRT_O --> LRT_SON
        
        %% Akış Bağlantıları
        LRT_A --> LRT_B --> LRT_C --> LRT_D --> LRT_E --> LRT_F --> LRT_G --> LRT_H
    end

    %% --- BÖLÜM 2: EĞİTİM (train_model) (Değerler Doldurulmuş) ---
    subgraph train_model ["Hibrit Zamanlayıcı Çakışma Akışı"]
        direction TB
        TR_A["Başla: (base_lr, max_lr) alınır"]
        TR_B["<b>Epoch Döngüsü</b> (epochs = 20*)"]
        
        TR_C["<b>Batch 0 (İlk Batch)</b><br><i>LR = base_lr* (veya Cosine'dan kalan)</i><br>autocast ile forward pass<br>loss = (sbp_mae + dbp_mae)/2"]
        TR_D["<b>Geri Yayılım (AMP):</b><br>scaler.scale(loss).backward()<br>scaler.step(optimizer)<br>scaler.update()"]
        TR_E["<b>2. CyclicLR.step() çağrılır</b><br><i>(LR'yi batch seviyesinde ezer)</i><br>mode='triangular2'<br>step_size_up=len(train_loader)//2*"]
        
        TR_F["<b>Batch 1 -> N Döngüsü</b>"]
        TR_G["<b>Batch Eğitimi (AMP):</b><br>zero_grad -> autocast -> loss -> <br>scaler.scale -> scaler.step -> scaler.update"]
        TR_H["<b>2. CyclicLR.step() çağrılır</b><br><i>(LR'yi batch seviyesinde ayarlamaya devam eder)</i>"]
        
        TR_I["Validation Döngüsü (model.eval)"]
        TR_J["<b>3. CosineAnnealingLR.step() çağrılır</b><br><i><b>Bir Sonraki</b> Epoch'un Batch 0'ı için LR'yi ayarlar</i><br>T_max=10, eta_min=1e-6"]
        TR_K{"<b>Early Stopping Kontrolü</b><br>if val_loss < best_val_loss - 1e-4* (delta)<br>patience_counter += 1<br>if patience_counter >= 15* (patience)"}
        
        TR_A --> TR_B
        TR_B -- Epoch başlar --> TR_C
        TR_C --> TR_D
        TR_D --> TR_E
        TR_E --> TR_F
        TR_F -- Her Kalan Batch için --> TR_G
        TR_G --> TR_H
        TR_H --> TR_F
        TR_F -- Döngü Bitti --> TR_I
        TR_I --> TR_J
        TR_J --> TR_K
        TR_K -- Devam --> TR_B
        TR_K -- Dur --> TR_SON[Model Eğitildi]
    end

    %% --- AÇIKLAMA KUTUSU (Akışa Bağlı Değil) - TEK KUTU HALİNE GETİRİLDİ ---
    InfoBox["<b>(*) İşaretli Değerler (Dinamik):</b><br>Bu değerler `train_model` fonksiyonuna<br>parametre olarak verilir veya dinamik olarak hesaplanır.<br><br><b>20* (epochs):</b> Toplam epoch sayısı (varsayılan=20)<br><b>base_lr*:</b> LR Test'ten gelen veya manuel girilen min LR<br><b>len(train_loader)//2*:</b> Eğitim verisi batch sayısının yarısı<br><b>1e-4* (delta):</b> En iyi loss'taki minimum iyileşme (düzeltildi)<br><b>15* (patience):</b> İyileşme olmazsa beklenecek epoch (varsayılan=15)<br><br><b>Sabit Değerler (Koddan):</b><br><b>beta (0.98):</b> LR Test kaybı için yumuşatma (smoothing) faktörü."]


    %% --- İki bölümü birbirine bağla ---
    LRT_SON --> TR_A

    %% --- Stil ---
    style LRT_F fill:#f99
    style TR_E fill:#cff
    style TR_J fill:#cff
    style InfoBox fill:#fef,stroke:#ccc,stroke-dasharray: 5 5 
```
