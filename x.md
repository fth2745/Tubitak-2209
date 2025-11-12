# Model Mimarisi Akış Şeması

Aşağıda `BP_TripleHybrid` modelinin `forward` metodunun görsel bir akışı bulunmaktadır.

```mermaid
flowchart LR
  subgraph Girdiler
    A[Baslat] --> B[clear_cuda]
    B --> C[HParam ve seed]
    C --> D[CIFAR10 veri kaynagi]
    D --> E[Veri artirma base v2]
    E --> F[Train ve Val loader]
  end

  subgraph LR_Finder
    direction LR
    F --> G[Model: CIFARResNet18]
    F --> H[LR Finder]
    G --> H
    H --> I[Max lr guncelle]
  end

  subgraph Egitim
    direction LR
    I --> J[Optimizer: SGD]
    J --> K[Scheduler: OneCycleLR]
    K --> L[Egitim dongusu]
    L --> M[train_one_epoch]
    M --> N[Degerlendirme]
    N --> O{Iyilesme?}
    O -- Evet --> P[Kaydet: %9491_best.pt]
    O -- Hayir --> Q[Patience += 1]
    Q --> R{Patience >= 15}
    R -- Hayir --> L
    R -- Evet --> S[Bitis]
    P --> L
  end

```
