using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.IO;
using System.Linq;

namespace BrainLLM;

/// <summary>
/// ニューロン状態スナップショット - Neuron全体を保存・復元するための構造体
/// 生物学的には: シナプス再現（Replay）のメモリ
/// </summary>
public class NeuronState
{
    public int NeuronId { get; set; }
    public float ActionPotential { get; set; }
    public float FiringHistory { get; set; }
    public Dictionary<int, float> DendriteWeights { get; set; } = new();  // SourceId → Weight
    public Dictionary<int, float> AxonWeights { get; set; } = new();      // TargetId → Weight
    public Dictionary<int, float> SaltatoryConductionWeights { get; set; } = new();
    public long Timestamp { get; set; }  // 記憶が記録された時刻
    
    /// <summary>
    /// ニューロン状態のディープコピー
    /// </summary>
    public NeuronState Clone()
    {
        return new NeuronState
        {
            NeuronId = this.NeuronId,
            ActionPotential = this.ActionPotential,
            FiringHistory = this.FiringHistory,
            DendriteWeights = new Dictionary<int, float>(this.DendriteWeights),
            AxonWeights = new Dictionary<int, float>(this.AxonWeights),
            SaltatoryConductionWeights = new Dictionary<int, float>(this.SaltatoryConductionWeights),
            Timestamp = this.Timestamp
        };
    }
}

/// <summary>
/// デンドライト（樹状突起）- 入力端子
/// </summary>
public class Dendrite
{
    public float Weight { get; set; }
    public float Value { get; set; }
    public int SourceNeuronId { get; set; }

    public Dendrite(int sourceNeuronId, float initialWeight = 0.1f)
    {
        SourceNeuronId = sourceNeuronId;
        Weight = initialWeight;
        Value = 0;
    }

    public float ComputeSignal() => Value * Weight;
}

/// <summary>
/// ソーマ（細胞体）- ニューロンの計算中心
/// </summary>
public class Soma
{
    public float Threshold { get; set; } = 0f;
    public float RestingPotential { get; set; } = -0.07f;
    private float membranePotential;

    public Soma()
    {
        membranePotential = RestingPotential;
    }

    public float Integrate(List<float> dendriteSignals)
    {
        // シナプス後電位の加算（重み付き入力の合計）
        float totalSignal = dendriteSignals.Sum();
        
        // ReLU活性化関数として動作（ニューラルネットワーク的）
        float output = Math.Max(0, totalSignal);
        
        // 膜電位を更新（将来の学習のため保持）
        membranePotential = output;

        return output;
    }
}

/// <summary>
/// 軸索ターミナル（シナプス末端）- 出力分岐
/// </summary>
public class AxonTerminal
{
    public int TargetNeuronId { get; set; }
    public float SynapticWeight { get; set; }
    public float NeurotransmitterLevel { get; set; } = 0;

    public AxonTerminal(int targetNeuronId, float initialWeight = 0.1f)
    {
        TargetNeuronId = targetNeuronId;
        SynapticWeight = initialWeight;
    }

    public void Release(float actionPotential)
    {
        // 活動電位がシナプス小胞の放出をトリガー
        NeurotransmitterLevel = actionPotential * SynapticWeight;
    }
}

/// <summary>
/// ニューロン（神経細胞）- 脳細胞の基本単位
/// </summary>
public class Neuron
{
    public int Id { get; }
    public string Name { get; set; }

    // 細胞部分
    private Soma soma;
    private List<Dendrite> dendrites = new();
    private List<AxonTerminal> axonTerminals = new();

    // 活動
    public float ActionPotential { get; set; } = 0;
    private float firingHistory = 0;
    
    // 状態メモリ（Neuron全体のスナップショット）
    private Queue<NeuronState> stateMemory;
    private int memoryCapacity;
    private int currentTimeStep = 0;
    
    // 跳躍伝導用の遠方軸索
    private List<SaltatoryConductionTerminal> saltatoryConductionAxons = new();

    public Neuron(int id, string name = "", int memoryCapacity = 10)
    {
        Id = id;
        Name = name ?? $"Neuron_{id}";
        soma = new Soma();
        this.memoryCapacity = memoryCapacity;
        stateMemory = new Queue<NeuronState>(memoryCapacity);
    }

    /// <summary>
    /// 現在のニューロン状態をスナップショットとして保存
    /// </summary>
    public void SaveState()
    {
        var state = new NeuronState
        {
            NeuronId = this.Id,
            ActionPotential = this.ActionPotential,
            FiringHistory = this.firingHistory,
            Timestamp = DateTime.UtcNow.Ticks
        };
        
        // デンドライトの重みをコピー
        foreach (var dendrite in dendrites)
        {
            state.DendriteWeights[dendrite.SourceNeuronId] = dendrite.Weight;
        }
        
        // 軸索の重みをコピー
        foreach (var axon in axonTerminals)
        {
            state.AxonWeights[axon.TargetNeuronId] = axon.SynapticWeight;
        }
        
        // 跳躍伝導の重みをコピー
        foreach (var saltatory in saltatoryConductionAxons)
        {
            state.SaltatoryConductionWeights[saltatory.TargetNeuronId] = saltatory.ConductionStrength;
        }
        
        stateMemory.Enqueue(state);
        
        // メモリ容量を超えた場合は古いものを削除
        if (stateMemory.Count > memoryCapacity)
        {
            stateMemory.Dequeue();
        }
        
        currentTimeStep++;
    }
    
    /// <summary>
    /// 過去のニューロン状態を復元（時間ステップ指定）
    /// stepsAgo=1: 最新, stepsAgo=2: 1ステップ前, etc.
    /// </summary>
    public bool RestoreState(int stepsAgo)
    {
        if (stepsAgo <= 0 || stepsAgo > stateMemory.Count)
            return false;
        
        var states = stateMemory.ToArray();
        // 配列の最後が最新（最後に追加された要素）
        int targetIndex = states.Length - stepsAgo;
        if (targetIndex < 0)
            return false;
            
        var targetState = states[targetIndex];
        
        this.ActionPotential = targetState.ActionPotential;
        this.firingHistory = targetState.FiringHistory;
        
        // デンドライトの重みを復元
        foreach (var dendrite in dendrites)
        {
            if (targetState.DendriteWeights.TryGetValue(dendrite.SourceNeuronId, out float weight))
            {
                dendrite.Weight = weight;
            }
        }
        
        // 軸索の重みを復元
        foreach (var axon in axonTerminals)
        {
            if (targetState.AxonWeights.TryGetValue(axon.TargetNeuronId, out float weight))
            {
                axon.SynapticWeight = weight;
            }
        }
        
        // 跳躍伝導の重みを復元
        foreach (var saltatory in saltatoryConductionAxons)
        {
            if (targetState.SaltatoryConductionWeights.TryGetValue(saltatory.TargetNeuronId, out float weight))
            {
                saltatory.ConductionStrength = weight;
            }
        }
        
        return true;
    }
    
    /// <summary>
    /// 記憶を参照して影響を取得（正規化版）
    /// </summary>
    public float GetMemoryInfluence()
    {
        if (stateMemory.Count == 0) return 0;
        
        // 過去の活動電位から加重平均を計算
        float influence = 0;
        float totalWeight = 0;
        
        var states = stateMemory.ToArray();
        for (int i = 0; i < states.Length; i++)
        {
            // 最近のほうが強い重み（指数分布）
            float weight = (float)Math.Exp(i / (float)states.Length);
            influence += states[i].ActionPotential * weight;
            totalWeight += weight;
        }
        
        // 正規化
        return totalWeight > 0 ? influence / totalWeight : 0;
    }
    
    /// <summary>
    /// 記憶バッファを取得
    /// </summary>
    public Queue<NeuronState> GetStateMemory() => new(stateMemory);

    /// <summary>
    /// デンドライト（入力）を追加
    /// </summary>
    public void AddDendrite(int sourceNeuronId, float initialWeight = 0.1f)
    {
        dendrites.Add(new Dendrite(sourceNeuronId, initialWeight));
    }

    /// <summary>
    /// 軸索ターミナル（出力分岐）を追加
    /// </summary>
    public void AddAxonTerminal(int targetNeuronId, float initialWeight = 0.1f)
    {
        axonTerminals.Add(new AxonTerminal(targetNeuronId, initialWeight));
    }
    
    /// <summary>
    /// 跳躍伝導軸索を追加（複数層をスキップして信号伝達）
    /// </summary>
    public void AddSaltatoryConductionAxon(int targetNeuronId, float conductionStrength = 0.1f)
    {
        saltatoryConductionAxons.Add(new SaltatoryConductionTerminal
        {
            TargetNeuronId = targetNeuronId,
            ConductionStrength = conductionStrength,
            LastActivityTime = -100
        });
    }

    /// <summary>
    /// 入力信号を受け取る
    /// </summary>
    public void ReceiveSignal(int dendriteIndex, float value)
    {
        if (dendriteIndex >= 0 && dendriteIndex < dendrites.Count)
        {
            dendrites[dendriteIndex].Value = value;
        }
    }

    /// <summary>
    /// ニューロンの活動を計算（Forward pass）
    /// </summary>
    public void Fire()
    {
        // デンドライトの信号を計算
        var dendriteSignals = dendrites.Select(d => d.ComputeSignal()).ToList();
        
        // 記憶（過去の状態）からの影響を加味
        float memoryInfluence = GetMemoryInfluence();
        
        // ソーマで統合
        float baseSignal = soma.Integrate(dendriteSignals);
        ActionPotential = baseSignal + memoryInfluence * 0.1f;  // 10%の記憶影響

        // 発火履歴を保存（学習用）
        firingHistory = firingHistory * 0.9f + ActionPotential * 0.1f;

        // 軸索ターミナルで神経伝達物質を放出
        foreach (var terminal in axonTerminals)
        {
            terminal.Release(ActionPotential);
        }
        
        // 跳躍伝導軸索から出力
        foreach (var saltatory in saltatoryConductionAxons)
        {
            saltatory.LastActivityTime = currentTimeStep;
            saltatory.Conduct(ActionPotential);
        }
        
        // **現在の状態をメモリに保存**
        SaveState();
    }

    /// <summary>
    /// 軸索ターミナルから出力を取得
    /// </summary>
    public float GetAxonOutput(int terminalIndex)
    {
        if (terminalIndex >= 0 && terminalIndex < axonTerminals.Count)
        {
            return axonTerminals[terminalIndex].NeurotransmitterLevel;
        }
        return 0;
    }

    /// <summary>
    /// シナプス可塑性（STDP - Spike-Timing-Dependent Plasticity）
    /// </summary>
    public void ApplySTDP(int dendriteIndex, float presynapticActivity, float postsynapticActivity, float learningRate = 0.01f)
    {
        if (dendriteIndex >= 0 && dendriteIndex < dendrites.Count)
        {
            float hebbian = presynapticActivity * postsynapticActivity;
            dendrites[dendriteIndex].Weight += learningRate * hebbian;

            // 重みクリップ [0.01, 1.0]
            dendrites[dendriteIndex].Weight = Math.Max(0.01f, Math.Min(1.0f, dendrites[dendriteIndex].Weight));
        }
    }

    /// <summary>
    /// 重みの勾配を計算（バックプロップ）
    /// </summary>
    public void UpdateWeights(float learningRate = 0.01f)
    {
        foreach (var dendrite in dendrites)
        {
            // 簡略化された学習ルール
            dendrite.Weight += learningRate * ActionPotential * dendrite.Value;
            dendrite.Weight = Math.Max(0.01f, Math.Min(1.0f, dendrite.Weight));
        }

        foreach (var terminal in axonTerminals)
        {
            terminal.SynapticWeight += learningRate * ActionPotential;
            terminal.SynapticWeight = Math.Max(0.01f, Math.Min(1.0f, terminal.SynapticWeight));
        }
    }

    /// <summary>
    /// デンドライトの重みを調整（バックプロップ用）
    /// </summary>
    public void AdjustDendriteWeight(int sourceNeuronId, float delta)
    {
        var dendrite = dendrites.FirstOrDefault(d => d.SourceNeuronId == sourceNeuronId);
        if (dendrite != null)
        {
            dendrite.Weight += delta;
            dendrite.Weight = Math.Max(-1.0f, Math.Min(1.0f, dendrite.Weight));
        }
    }

    /// <summary>
    /// ソースIDでデンドライトを取得
    /// </summary>
    public Dendrite? GetDendrite(int sourceNeuronId)
    {
        return dendrites.FirstOrDefault(d => d.SourceNeuronId == sourceNeuronId);
    }

    /// <summary>
    /// すべてのデンドライトを取得
    /// </summary>
    public List<Dendrite> GetDendrites() => dendrites;

    /// <summary>
    /// すべての軸索ターミナルを取得
    /// </summary>
    public List<AxonTerminal> GetAxonTerminals() => axonTerminals;
    
    /// <summary>
    /// 跳躍伝導軸索を取得
    /// </summary>
    public List<SaltatoryConductionTerminal> GetSaltatoryConductionAxons() => saltatoryConductionAxons;

    public int DendriteCount => dendrites.Count;
    public int AxonTerminalCount => axonTerminals.Count;
    public float FiringHistory => firingHistory;
}

/// <summary>
/// 跳躍伝導ターミナル（複数層をスキップして信号伝達）
/// 生物学的には: 有髄軸索のランビエ絞輪を通じた効率的な伝導
/// </summary>
public class SaltatoryConductionTerminal
{
    public int TargetNeuronId { get; set; }
    public float ConductionStrength { get; set; }
    public float ConductionOutput { get; set; } = 0;
    public int LastActivityTime { get; set; }  // STDP用
    
    /// <summary>
    /// 活動電位を遠方へ伝導
    /// </summary>
    public void Conduct(float actionPotential)
    {
        // 跳躍伝導は速く、信号減衰が少ない
        ConductionOutput = actionPotential * ConductionStrength;
    }
    
    /// <summary>
    /// 伝導強度を学習
    /// </summary>
    public void UpdateConductionStrength(float delta)
    {
        ConductionStrength += delta;
        ConductionStrength = Math.Max(0.01f, Math.Min(1.0f, ConductionStrength));
    }
}

/// <summary>
/// ニューロンシリアライゼーション拡張 - JSON永続化
/// </summary>
public static class NeuronSerializer
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never
    };

    /// <summary>
    /// 単一ニューロンをJSONファイルに保存
    /// </summary>
    public static void SaveToFile(this Neuron neuron, string filePath)
    {
        var state = new NeuronState
        {
            NeuronId = neuron.Id,
            ActionPotential = neuron.ActionPotential,
            FiringHistory = neuron.FiringHistory,
            Timestamp = DateTime.UtcNow.Ticks
        };
        
        // 全ての接続情報を保存
        foreach (var dendrite in neuron.GetDendrites())
        {
            state.DendriteWeights[dendrite.SourceNeuronId] = dendrite.Weight;
        }
        
        foreach (var axon in neuron.GetAxonTerminals())
        {
            state.AxonWeights[axon.TargetNeuronId] = axon.SynapticWeight;
        }
        
        foreach (var saltatory in neuron.GetSaltatoryConductionAxons())
        {
            state.SaltatoryConductionWeights[saltatory.TargetNeuronId] = saltatory.ConductionStrength;
        }
        
        string json = JsonSerializer.Serialize(state, JsonOptions);
        File.WriteAllText(filePath, json);
    }

    /// <summary>
    /// JSONからニューロン状態を読み込み（新しいNeuronインスタンスは作成しない、既存のニューロンに適用）
    /// </summary>
    public static NeuronState LoadStateFromFile(string filePath)
    {
        string json = File.ReadAllText(filePath);
        return JsonSerializer.Deserialize<NeuronState>(json, JsonOptions)
            ?? throw new InvalidOperationException("Failed to deserialize neuron state");
    }
}
