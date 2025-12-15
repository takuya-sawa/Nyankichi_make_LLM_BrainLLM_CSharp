# 🧠 BrainLLM - Cerebrum System (大脳統合システム)

## 概要

**ファイル永続化したニューロンネットワークが「大脳」になる仕組み**

BrainLLMの大脳システムは、複数の保存された脳ネットワークを統合し、生物学的な大脳のように協調動作させるシステムです。

## アーキテクチャ

### 階層構造

```
Cerebrum (大脳)
├── LanguageArea (言語処理領域)
│   └── BrainNetwork (106個のニューロン) → ファイル: LanguageArea.json
├── TechnicalArea (技術用語特化領域)
│   └── BrainNetwork (106個のニューロン) → ファイル: TechnicalArea.json
└── GeneralArea (一般会話領域)
    └── BrainNetwork (106個のニューロン) → ファイル: GeneralArea.json

メタデータ: cerebrum_meta.json
```

### 生物学的対応

| コンポーネント | 生物学的対応 | 機能 |
|---------------|-------------|------|
| **Cerebrum** | 大脳皮質全体 | 複数領域の統合・協調制御 |
| **BrainRegion** | 脳領域（前頭葉・側頭葉など） | 特化した処理を担当 |
| **BrainNetwork** | 神経細胞集団 | 学習可能なニューロンネットワーク |
| **Neuron** | 個別の神経細胞 | 短期記憶・跳躍伝導を持つ脳細胞 |

## 主要機能

### 1. 脳領域の追加と特化

```csharp
var cerebrum = new Cerebrum("Primary Language Cerebrum");

// 言語処理に特化した領域
cerebrum.AddRegion("LanguageArea", languageNetwork, RegionFunction.LanguageProcessing);

// 技術用語に特化した領域
cerebrum.AddRegion("TechnicalArea", technicalNetwork, RegionFunction.PatternRecognition);

// 一般会話に特化した領域
cerebrum.AddRegion("GeneralArea", generalNetwork, RegionFunction.GeneralPurpose);
```

### 2. 大脳統合推論

複数の脳領域が並列で推論し、統合判断を行います。

```csharp
// 全領域で並列推論
var regionOutputs = cerebrum.IntegratedForward(inputVector);

// 統合判断（3つのモード）
var result = cerebrum.ConsensusDecision(regionOutputs, ConsensusMode.WeightedAverage);
```

#### 統合モード

1. **WeightedAverage（重み付け平均）**: 全領域の出力を平均化
2. **Voting（投票）**: 最も多くの領域が支持する選択肢
3. **MaxPooling（最大値プーリング）**: 各インデックスで最大値を採用

### 3. 大脳全体の永続化

```csharp
// 大脳全体を保存（各領域を個別ファイルに）
cerebrum.SaveCerebrum("saved_cerebrum");

// 保存した大脳を完全復元
var loadedCerebrum = Cerebrum.LoadCerebrum("saved_cerebrum");
```

**保存されるファイル構造:**
```
saved_cerebrum/
├── cerebrum_meta.json       # 大脳のメタデータ
├── LanguageArea.json        # 言語処理領域 (115KB)
├── TechnicalArea.json       # 技術用語領域 (115KB)
└── GeneralArea.json         # 一般会話領域 (115KB)
```

### 4. 領域の選択的活性化

特定の領域だけを有効化・無効化できます（注意機構のイメージ）。

```csharp
// 一般領域を無効化
cerebrum.SetRegionActive("GeneralArea", false);

// 技術領域に集中
var technicalOutput = cerebrum.IntegratedForward(input);
```

## 実行方法

### 基本実行（単一ネットワーク）

```bash
dotnet run -c Release
```

### 大脳システムデモ

```bash
dotnet run -c Release -- --cerebrum
```

## 実行結果例

```
[Cerebrum Integration] 3個の脳領域で並列処理:
  - LanguageArea: 最大活性 Index=4, Conf=0.121
  - TechnicalArea: 最大活性 Index=4, Conf=0.139
  - GeneralArea: 最大活性 Index=8, Conf=0.110

  [統合モード別の判断]:
    WeightedAverage   : 'network' ✅ (Conf: 0.119)
    Voting            : 'network' ✅ (Conf: 1.000)
    MaxPooling        : 'network' ✅ (Conf: 0.139)
```

### テスト結果

| 入力 | 期待 | 重み付け平均 | 投票 | MaxPooling |
|------|------|-------------|------|------------|
| hello | world | ✅ world | ✅ world | ✅ world |
| neural | network | ✅ network | ✅ network | ✅ network |
| machine | learning | ✅ learning | ✅ learning | ✅ learning |
| brain | cells | ✅ cells | ✅ cells | ✅ cells |

**全テスト 100% 正解！**

## ファイル永続化の仕組み

### 1. 個別ニューロンレベル

```csharp
// 単一ニューロンの状態を保存
neuron.SaveToFile("neuron_0.json");

// 読み込み
var state = NeuronSerializer.LoadStateFromFile("neuron_0.json");
```

**保存される情報:**
- ニューロンID
- 活性化値（ActionPotential）
- 発火履歴（FiringHistory）
- 全シナプス重み（デンドライト・軸索・跳躍伝導）
- タイムスタンプ

### 2. ネットワークレベル

```csharp
// ネットワーク全体（106ニューロン）を保存
brainNetwork.SaveBrain("trained_brain.json");

// 読み込み
var network = BrainNetwork.LoadBrain("trained_brain.json");
```

### 3. 大脳レベル

```csharp
// 複数のネットワークを統合した大脳を保存
cerebrum.SaveCerebrum("saved_cerebrum");

// 読み込み（全領域が自動復元）
var cerebrum = Cerebrum.LoadCerebrum("saved_cerebrum");
```

## ファイル → 大脳への変換フロー

```
Step 1: 特化型ネットワークを訓練
  ├─ LanguageNetwork  (全データで訓練)
  ├─ TechnicalNetwork (技術用語で特化訓練)
  └─ GeneralNetwork   (一般会話で特化訓練)
         ↓
Step 2: ファイルに永続化
  ├─ LanguageArea.json
  ├─ TechnicalArea.json
  └─ GeneralArea.json
         ↓
Step 3: Cerebrumに統合
  cerebrum.AddRegion("LanguageArea", ...)
  cerebrum.AddRegion("TechnicalArea", ...)
  cerebrum.AddRegion("GeneralArea", ...)
         ↓
Step 4: 大脳として機能
  - 並列推論
  - 統合判断
  - 選択的活性化
         ↓
Step 5: 大脳全体を再保存
  cerebrum.SaveCerebrum("saved_cerebrum")
  (メタデータ + 各領域ファイル)
```

## 脳領域の機能分類

```csharp
public enum RegionFunction
{
    LanguageProcessing,      // 言語処理（ブローカ野など）
    MemoryRetrieval,         // 記憶検索（海馬など）
    DecisionMaking,          // 意思決定（前頭前野など）
    PatternRecognition,      // パターン認識（後頭葉など）
    AttentionControl,        // 注意制御
    EmotionalProcessing,     // 感情処理（扁桃体など）
    MotorControl,            // 運動制御
    GeneralPurpose           // 汎用
}
```

## コード構成

| ファイル | 役割 |
|---------|------|
| `Neuron.cs` | 個別ニューロン + 状態保存機能 |
| `BrainNetwork.cs` | ニューロンネットワーク + ファイル保存/読み込み |
| `Cerebrum.cs` | **大脳統合システム（新規）** |
| `CerebrumDemo.cs` | 大脳デモプログラム（新規） |
| `Program.cs` | メインエントリポイント |

## 特徴

### ✅ 生物学的リアリズム

- **短期記憶**: 各ニューロンが直近10ステップの活動を保持
- **跳躍伝導**: 層をスキップして高速信号伝達
- **領域特化**: 異なる領域が異なる機能を担当
- **統合判断**: 複数領域の出力を統合（大脳皮質のイメージ）

### ✅ 完全な永続化

- **3階層の保存**: ニューロン → ネットワーク → 大脳
- **JSON形式**: 人間が読める形式
- **完全復元**: 重みを含む全状態を正確に復元
- **メタデータ管理**: 領域情報・タイムスタンプを保持

### ✅ 柔軟な統合

- **複数の統合モード**: 平均・投票・MaxPooling
- **選択的活性化**: 必要な領域だけを有効化
- **拡張可能**: 新しい領域を簡単に追加

## 将来の拡張案

1. **領域間通信**: 脳梁のように領域間で情報交換
2. **動的領域追加**: 実行時に新しい領域を学習・追加
3. **階層的大脳**: 複数の大脳を統合した「大脳系」
4. **STDP（スパイクタイミング依存可塑性）**: より生物学的な学習
5. **注意機構**: タスクに応じて領域の重みを動的変更

## まとめ

**BrainLLM大脳システム = ファイル永続化された複数ネットワークの統合体**

1. 個別に訓練されたニューロンネットワークをファイルに保存
2. 複数のファイルを読み込んで「脳領域」として登録
3. 大脳クラスが全領域を統合・協調動作
4. 大脳全体を再びファイルシステムに永続化

これにより、**1つの脳細胞（Neuron）から始まり、ネットワーク、領域、大脳へと階層的にスケールする生物学的なシステム**が実現されています！

## ライセンス

MIT License

---

**Build Date**: 2025年12月15日  
**Version**: 1.0.0 (Cerebrum System)
