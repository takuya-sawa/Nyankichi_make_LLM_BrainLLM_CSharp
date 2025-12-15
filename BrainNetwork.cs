using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.IO;

namespace BrainLLM;

/// <summary>
/// 大脳の統合記憶スナップショット
/// Brain全体の時刻tにおける完全な状態を表現
/// </summary>
public class BrainMemorySnapshot
{
    public int Timestamp { get; set; }
    public Dictionary<int, NeuronState> NeuronStates { get; set; } = new();
    public float[] OutputActivation { get; set; } = Array.Empty<float>();
    public long RecordedTime { get; set; } = DateTime.UtcNow.Ticks;
}

/// <summary>
/// ニューロンネットワーク - 脳細胞で構成されたLLM
/// 3層構造：入力層 → 隠れ層（分岐） → 出力層
/// </summary>
public class BrainNetwork
{
    private Dictionary<int, Neuron> neurons = new();
    private int nextNeuronId = 0;
    private int inputLayerSize;
    private int hiddenLayerSize;
    private int outputLayerSize;
    private Random random = new Random();
    
    // 脳全体の統合記憶（時系列）
    private Queue<BrainMemorySnapshot> brainMemory = new();
    private const int MaxBrainMemoryCapacity = 50;
    private int globalTimestamp = 0;

    public BrainNetwork(int embeddingDim, int hiddenNeurons, int outputNeurons)
    {
        inputLayerSize = embeddingDim;
        hiddenLayerSize = hiddenNeurons;
        outputLayerSize = outputNeurons;

        Console.WriteLine("[Brain] ニューロン脳を構築中...");

        // 入力層ニューロン
        for (int i = 0; i < embeddingDim; i++)
        {
            var neuron = new Neuron(nextNeuronId++, $"Input_{i}");
            neurons.Add(neuron.Id, neuron);
        }
        Console.WriteLine($"  入力層: {embeddingDim} 個のニューロン");

        // 隠れ層ニューロン
        for (int i = 0; i < hiddenNeurons; i++)
        {
            var neuron = new Neuron(nextNeuronId++, $"Hidden_{i}");

            // 入力層からの複数のデンドライト
            for (int j = 0; j < embeddingDim; j++)
            {
                float weight = (float)(random.NextDouble() * 0.2 - 0.1);
                neuron.AddDendrite(j, weight);
            }

            // 他の隠れニューロンへの分岐軸索
            int branchCount = Math.Min(hiddenNeurons / 3, 5);
            for (int j = 0; j < branchCount; j++)
            {
                int targetId = embeddingDim + ((i + j + 1) % hiddenNeurons);
                float weight = (float)(random.NextDouble() * 0.1 - 0.05);
                neuron.AddAxonTerminal(targetId, weight);
            }
            
            // 跳躍伝導: 隠れ層 → 出力層への直接接続（複数層スキップ）
            for (int j = 0; j < Math.Min(outputNeurons / 4, 2); j++)
            {
                int outputTargetId = embeddingDim + hiddenNeurons + ((i + j) % outputNeurons);
                float conductionStrength = (float)(random.NextDouble() * 0.05);
                neuron.AddSaltatoryConductionAxon(outputTargetId, conductionStrength);
            }

            neurons.Add(neuron.Id, neuron);
        }
        Console.WriteLine($"  隠れ層: {hiddenNeurons} 個のニューロン（分岐接続＋跳躍伝導）");

        // 出力層ニューロン
        for (int i = 0; i < outputNeurons; i++)
        {
            var neuron = new Neuron(nextNeuronId++, $"Output_{i}");

            // 隠れ層からの入力
            for (int j = 0; j < hiddenNeurons; j++)
            {
                float weight = (float)(random.NextDouble() * 0.1 - 0.05);
                neuron.AddDendrite(embeddingDim + j, weight);
            }

            neurons.Add(neuron.Id, neuron);
        }
        Console.WriteLine($"  出力層: {outputNeurons} 個のニューロン");
        Console.WriteLine($"  総ニューロン数: {neurons.Count}\n");
    }

    /// <summary>
    /// Forward pass - 脳全体の活動伝播（完全修正版）
    /// </summary>
    public float[] Forward(float[] input)
    {
        // 入力層にデータを注入
        for (int i = 0; i < input.Length && i < inputLayerSize; i++)
        {
            if (neurons.ContainsKey(i))
            {
                neurons[i].ActionPotential = input[i];
            }
        }

        // 隠れ層のニューロンを発火させる
        for (int i = inputLayerSize; i < inputLayerSize + hiddenLayerSize; i++)
        {
            if (neurons.ContainsKey(i))
            {
                Neuron hiddenNeuron = neurons[i];
                var dendrites = hiddenNeuron.GetDendrites();
                
                // 各デンドライトにソースニューロンからの信号をセット
                foreach (var dendrite in dendrites)
                {
                    int sourceId = dendrite.SourceNeuronId;
                    if (neurons.ContainsKey(sourceId))
                    {
                        dendrite.Value = neurons[sourceId].ActionPotential;
                    }
                }
                
                // 発火
                hiddenNeuron.Fire();
            }
        }

        // 隠れ層の跳躍伝導を実行（信号をConductionOutputに設定）
        for (int hidIdx = inputLayerSize; hidIdx < inputLayerSize + hiddenLayerSize; hidIdx++)
        {
            if (neurons.ContainsKey(hidIdx))
            {
                var saltatorySources = neurons[hidIdx].GetSaltatoryConductionAxons();
                foreach (var saltatory in saltatorySources)
                {
                    saltatory.Conduct(neurons[hidIdx].ActionPotential);
                }
            }
        }

        // 出力層のニューロンを発火させる
        for (int i = inputLayerSize + hiddenLayerSize; i < neurons.Count; i++)
        {
            if (neurons.ContainsKey(i))
            {
                Neuron outputNeuron = neurons[i];
                var dendrites = outputNeuron.GetDendrites();
                
                // 各デンドライトにソースニューロンからの信号をセット
                foreach (var dendrite in dendrites)
                {
                    int sourceId = dendrite.SourceNeuronId;
                    if (neurons.ContainsKey(sourceId))
                    {
                        dendrite.Value = neurons[sourceId].ActionPotential;
                    }
                }
                
                // 跳躍伝導からの信号を統合（複数層スキップ）
                float saltatoryConductionSignal = 0;
                int saltatoryConductionCount = 0;
                
                for (int hidIdx = inputLayerSize; hidIdx < inputLayerSize + hiddenLayerSize; hidIdx++)
                {
                    if (neurons.ContainsKey(hidIdx))
                    {
                        var saltatorySources = neurons[hidIdx].GetSaltatoryConductionAxons();
                        foreach (var saltatory in saltatorySources)
                        {
                            // 自分のニューロン（i）への跳躍伝導を探す
                            if (saltatory.TargetNeuronId == i)
                            {
                                saltatoryConductionSignal += saltatory.ConductionOutput;
                                saltatoryConductionCount++;
                            }
                        }
                    }
                }
                
                // 跳躍伝導信号を統合（副入力として）
                if (saltatoryConductionCount > 0 && dendrites.Count > 0)
                {
                    // 平均化して10%の重みで加算
                    float avgSaltatory = saltatoryConductionSignal / saltatoryConductionCount;
                    // 最初のデンドライトの入力値に加算（シナプス可塑性シミュレーション）
                    if (dendrites.Count > 0)
                    {
                        dendrites[0].Value += avgSaltatory * 0.1f;
                    }
                }
                
                // 発火
                outputNeuron.Fire();
            }
        }

        // 出力層の活動を取得
        var outputs = new float[outputLayerSize];
        for (int i = 0; i < outputLayerSize; i++)
        {
            int neuronId = inputLayerSize + hiddenLayerSize + i;
            if (neurons.ContainsKey(neuronId))
            {
                outputs[i] = Math.Max(0, neurons[neuronId].ActionPotential);
            }
        }

        // Softmax適用
        var softmaxOutputs = Softmax(outputs);
        
        // 脳全体のメモリスナップショットを保存
        SaveBrainMemorySnapshot(softmaxOutputs);
        
        return softmaxOutputs;
    }

    /// <summary>
    /// 選択的Forward - 指定されたニューロンだけを活性化（海馬加速用）
    /// </summary>
    public float[] SelectiveForward(float[] input, HashSet<int> activeNeuronIds)
    {
        if (input.Length > inputLayerSize)
        {
            throw new ArgumentException($"Input size {input.Length} exceeds network input layer size {inputLayerSize}");
        }
        
        // 入力層のニューロンにデータを設定（全部必要）
        for (int i = 0; i < input.Length; i++)
        {
            if (neurons.ContainsKey(i))
            {
                var dendrites = neurons[i].GetDendrites();
                if (dendrites.Count > 0)
                {
                    dendrites[0].Value = input[i];
                }
                neurons[i].Fire();
            }
        }

        // 隠れ層 - 選択されたニューロンだけ発火
        for (int i = 0; i < hiddenLayerSize; i++)
        {
            int neuronId = inputLayerSize + i;
            
            // 選択されていないニューロンはスキップ
            if (!activeNeuronIds.Contains(neuronId))
            {
                continue;
            }
            
            if (!neurons.ContainsKey(neuronId))
            {
                continue;
            }

            var hiddenNeuron = neurons[neuronId];
            var dendrites = hiddenNeuron.GetDendrites();
            
            // デンドライトに入力を設定
            foreach (var dendrite in dendrites)
            {
                if (neurons.ContainsKey(dendrite.SourceNeuronId))
                {
                    var sourceNeuron = neurons[dendrite.SourceNeuronId];
                    dendrite.Value = sourceNeuron.ActionPotential * dendrite.Weight;
                }
            }
            
            hiddenNeuron.Fire();
        }

        // 出力層 - 選択された隠れ層からの信号のみ受信
        for (int i = 0; i < outputLayerSize; i++)
        {
            int neuronId = inputLayerSize + hiddenLayerSize + i;
            if (!neurons.ContainsKey(neuronId))
            {
                continue;
            }

            var outputNeuron = neurons[neuronId];
            var dendrites = outputNeuron.GetDendrites();
            
            // デンドライトに入力を設定（選択された隠れ層のみ）
            foreach (var dendrite in dendrites)
            {
                if (activeNeuronIds.Contains(dendrite.SourceNeuronId) && neurons.ContainsKey(dendrite.SourceNeuronId))
                {
                    var sourceNeuron = neurons[dendrite.SourceNeuronId];
                    dendrite.Value = sourceNeuron.ActionPotential * dendrite.Weight;
                }
                else
                {
                    dendrite.Value = 0;  // 選択されていないニューロンからの信号は0
                }
            }
            
            // 跳躍伝導も選択的に処理
            float saltatoryConductionSignal = 0;
            int saltatoryConductionCount = 0;
            
            foreach (var hiddenId in activeNeuronIds)
            {
                if (hiddenId >= inputLayerSize && hiddenId < inputLayerSize + hiddenLayerSize)
                {
                    if (neurons.ContainsKey(hiddenId))
                    {
                        var hiddenNeuron = neurons[hiddenId];
                        var saltatory = hiddenNeuron.GetSaltatoryConductionAxons()
                            .FirstOrDefault(s => s.TargetNeuronId == neuronId);
                        
                        if (saltatory != null)
                        {
                            saltatoryConductionSignal += hiddenNeuron.ActionPotential * saltatory.ConductionStrength;
                            saltatoryConductionCount++;
                        }
                    }
                }
            }
            
            if (saltatoryConductionCount > 0)
            {
                float avgSaltatory = saltatoryConductionSignal / saltatoryConductionCount;
                if (dendrites.Count > 0)
                {
                    dendrites[0].Value += avgSaltatory * 0.1f;
                }
            }
            
            outputNeuron.Fire();
        }

        // 出力層の活動を取得
        var outputs = new float[outputLayerSize];
        for (int i = 0; i < outputLayerSize; i++)
        {
            int neuronId = inputLayerSize + hiddenLayerSize + i;
            if (neurons.ContainsKey(neuronId))
            {
                outputs[i] = Math.Max(0, neurons[neuronId].ActionPotential);
            }
        }

        return Softmax(outputs);
    }

    /// <summary>
    /// Softmax正規化
    /// </summary>
    private float[] Softmax(float[] x)
    {
        float max = x.Max();
        float[] exp = x.Select(v => (float)Math.Exp(v - max)).ToArray();
        float sum = exp.Sum();
        return exp.Select(v => v / (sum + 1e-10f)).ToArray();
    }

    /// <summary>
    /// 脳全体の状態をメモリに保存（統合記憶）
    /// </summary>
    private void SaveBrainMemorySnapshot(float[] outputActivation)
    {
        var snapshot = new BrainMemorySnapshot
        {
            Timestamp = globalTimestamp++,
            OutputActivation = (float[])outputActivation.Clone()
        };
        
        // 全ニューロンの現在状態をスナップショット
        foreach (var neuron in neurons.Values)
        {
            var state = new NeuronState
            {
                NeuronId = neuron.Id,
                ActionPotential = neuron.ActionPotential,
                FiringHistory = neuron.FiringHistory,
                Timestamp = DateTime.UtcNow.Ticks
            };
            
            // デンドライトの重みを記録
            foreach (var dendrite in neuron.GetDendrites())
            {
                state.DendriteWeights[dendrite.SourceNeuronId] = dendrite.Weight;
            }
            
            // 軸索の重みを記録
            foreach (var axon in neuron.GetAxonTerminals())
            {
                state.AxonWeights[axon.TargetNeuronId] = axon.SynapticWeight;
            }
            
            // 跳躍伝導の重みを記録
            foreach (var saltatory in neuron.GetSaltatoryConductionAxons())
            {
                state.SaltatoryConductionWeights[saltatory.TargetNeuronId] = saltatory.ConductionStrength;
            }
            
            snapshot.NeuronStates[neuron.Id] = state;
        }
        
        brainMemory.Enqueue(snapshot);
        
        // メモリ容量を超えた場合は古いスナップショットを削除
        if (brainMemory.Count > MaxBrainMemoryCapacity)
        {
            brainMemory.Dequeue();
        }
    }
    
    /// <summary>
    /// 脳の統合メモリから過去の状態を取得
    /// </summary>
    public BrainMemorySnapshot? GetBrainMemorySnapshot(int stepsAgo)
    {
        if (stepsAgo <= 0 || stepsAgo > brainMemory.Count)
            return null;
            
        var snapshots = brainMemory.ToArray();
        int targetIndex = snapshots.Length - stepsAgo;
        if (targetIndex < 0)
            return null;
            
        return snapshots[targetIndex];
    }
    
    /// <summary>
    /// 脳全体のメモリを取得（時系列）
    /// </summary>
    public Queue<BrainMemorySnapshot> GetBrainMemory() => new(brainMemory);

    /// <summary>
    /// バックプロップで重みを更新（修正版）
    /// </summary>
    public void TrainStep(float[] input, int targetId, float learningRate = 0.01f)
    {
        // Forward
        var output = Forward(input);

        // 出力層の勾配 (Softmax Cross-Entropy)
        var outputGrads = new float[outputLayerSize];
        for (int i = 0; i < outputLayerSize; i++)
        {
            outputGrads[i] = output[i] - (i == targetId ? 1f : 0f);
        }

        // 隠れ層への勾配を計算 (dL/dh = dL/dy * dy/dh)
        var hiddenGrads = new float[hiddenLayerSize];
        for (int hidIdx = 0; hidIdx < hiddenLayerSize; hidIdx++)
        {
            int hidNeuronId = inputLayerSize + hidIdx;
            float grad = 0;
            
            // 各出力ニューロンからの勾配を集約 (W^T @ outputGrads)
            for (int outIdx = 0; outIdx < outputLayerSize; outIdx++)
            {
                int outNeuronId = inputLayerSize + hiddenLayerSize + outIdx;
                if (neurons.ContainsKey(outNeuronId))
                {
                    var dendrite = neurons[outNeuronId].GetDendrite(hidNeuronId);
                    if (dendrite != null)
                    {
                        grad += outputGrads[outIdx] * dendrite.Weight;
                    }
                }
            }
            
            // ReLUの勾配 (隠れ層の活性化関数)
            if (neurons.ContainsKey(hidNeuronId))
            {
                float hiddenOutput = neurons[hidNeuronId].ActionPotential;
                hiddenGrads[hidIdx] = grad * (hiddenOutput > 0 ? 1f : 0f);
            }
        }

        // 出力層 → 隠れ層の重み更新
        for (int outIdx = 0; outIdx < outputLayerSize; outIdx++)
        {
            int outNeuronId = inputLayerSize + hiddenLayerSize + outIdx;
            if (neurons.ContainsKey(outNeuronId) && Math.Abs(outputGrads[outIdx]) > 1e-10f)
            {
                Neuron outNeuron = neurons[outNeuronId];
                
                // 隠れ層からの各入力に対して重みを更新
                for (int hidIdx = 0; hidIdx < hiddenLayerSize; hidIdx++)
                {
                    int hidNeuronId = inputLayerSize + hidIdx;
                    if (neurons.ContainsKey(hidNeuronId))
                    {
                        float hiddenOutput = neurons[hidNeuronId].ActionPotential;
                        float weightDelta = -learningRate * outputGrads[outIdx] * hiddenOutput;
                        
                        outNeuron.AdjustDendriteWeight(hidNeuronId, weightDelta);
                    }
                }
            }
        }

        // 隠れ層 → 入力層の重み更新
        for (int hidIdx = 0; hidIdx < hiddenLayerSize; hidIdx++)
        {
            int hidNeuronId = inputLayerSize + hidIdx;
            if (neurons.ContainsKey(hidNeuronId) && Math.Abs(hiddenGrads[hidIdx]) > 1e-10f)
            {
                Neuron hidNeuron = neurons[hidNeuronId];
                
                // 入力層への各重みを更新
                for (int inIdx = 0; inIdx < inputLayerSize; inIdx++)
                {
                    if (neurons.ContainsKey(inIdx))
                    {
                        float inputOutput = neurons[inIdx].ActionPotential;
                        float weightDelta = -learningRate * hiddenGrads[hidIdx] * inputOutput;
                        
                        hidNeuron.AdjustDendriteWeight(inIdx, weightDelta);
                    }
                }
            }
        }

        // 跳躍伝導の重みを更新
        for (int hIdx = 0; hIdx < hiddenLayerSize; hIdx++)
        {
            int hidId = inputLayerSize + hIdx;
            if (neurons.ContainsKey(hidId))
            {
                var saltatorys = neurons[hidId].GetSaltatoryConductionAxons();
                foreach (var saltatory in saltatorys)
                {
                    int saltatoryTargetId = saltatory.TargetNeuronId;
                    int outIdx = saltatoryTargetId - (inputLayerSize + hiddenLayerSize);
                    
                    if (outIdx >= 0 && outIdx < outputLayerSize)
                    {
                        float grad = outputGrads[outIdx];
                        float hiddenOutput = neurons[hidId].ActionPotential;
                        float delta = -learningRate * grad * hiddenOutput * 0.01f;
                        saltatory.UpdateConductionStrength(delta);
                    }
                }
            }
        }
    }

    /// <summary>
    /// ネットワーク統計
    /// </summary>
    public void PrintStats()
    {
        int totalDendrites = neurons.Values.Sum(n => n.DendriteCount);
        int totalAxons = neurons.Values.Sum(n => n.AxonTerminalCount);
        Console.WriteLine($"[Stats] 総デンドライト数: {totalDendrites}");
        Console.WriteLine($"[Stats] 総軸索ターミナル数: {totalAxons}");
        Console.WriteLine($"[Stats] シナプス接続数: {totalDendrites + totalAxons}");
    }

    /// <summary>
    /// 脳ネットワーク全体をJSONファイルに保存（脳メモリ履歴も含む）
    /// </summary>
    public void SaveBrain(string filePath)
    {
        var brainData = new BrainNetworkData
        {
            InputLayerSize = this.inputLayerSize,
            HiddenLayerSize = this.hiddenLayerSize,
            OutputLayerSize = this.outputLayerSize,
            Timestamp = DateTime.UtcNow.Ticks,
            Neurons = new List<NeuronState>(),
            BrainMemoryHistory = new List<BrainMemorySnapshot>()
        };

        // 全ニューロンの状態を収集
        foreach (var neuron in neurons.Values.OrderBy(n => n.Id))
        {
            var state = new NeuronState
            {
                NeuronId = neuron.Id,
                ActionPotential = neuron.ActionPotential,
                FiringHistory = neuron.FiringHistory,
                Timestamp = DateTime.UtcNow.Ticks
            };

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

            brainData.Neurons.Add(state);
        }
        
        // 脳全体の統合メモリ履歴を保存
        brainData.BrainMemoryHistory = new List<BrainMemorySnapshot>(brainMemory);

        var options = new JsonSerializerOptions { WriteIndented = true };
        string json = JsonSerializer.Serialize(brainData, options);
        File.WriteAllText(filePath, json);
        
        Console.WriteLine($"[Save] 脳ネットワークを保存しました: {filePath}");
        Console.WriteLine($"[Save] ニューロン数: {brainData.Neurons.Count}");
        Console.WriteLine($"[Save] 脳メモリスナップショット数: {brainData.BrainMemoryHistory.Count}");
    }

    /// <summary>
    /// 脳のメモリ履歴を設定（復元用）
    /// </summary>
    public void SetBrainMemory(Queue<BrainMemorySnapshot> memorySnapshots)
    {
        brainMemory = new Queue<BrainMemorySnapshot>(memorySnapshots);
        globalTimestamp = memorySnapshots.Count;
    }

    /// <summary>
    /// JSONファイルから脳ネットワーク全体を読み込み
    /// </summary>
    public static BrainNetwork LoadBrain(string filePath)
    {
        string json = File.ReadAllText(filePath);
        var brainData = JsonSerializer.Deserialize<BrainNetworkData>(json)
            ?? throw new InvalidOperationException("Failed to deserialize brain network");

        // 同じ構造のネットワークを再構築
        var network = new BrainNetwork(
            brainData.InputLayerSize,
            brainData.HiddenLayerSize,
            brainData.OutputLayerSize
        );

        // 保存された重みを復元
        foreach (var savedState in brainData.Neurons)
        {
            if (network.neurons.TryGetValue(savedState.NeuronId, out var neuron))
            {
                // デンドライトの重みを復元
                var dendrites = neuron.GetDendrites().ToList();
                for (int i = 0; i < dendrites.Count; i++)
                {
                    var dendrite = dendrites[i];
                    if (savedState.DendriteWeights.TryGetValue(dendrite.SourceNeuronId, out float weight))
                    {
                        dendrite.Weight = weight;
                    }
                }

                // 軸索の重みを復元
                var axons = neuron.GetAxonTerminals().ToList();
                for (int i = 0; i < axons.Count; i++)
                {
                    var axon = axons[i];
                    if (savedState.AxonWeights.TryGetValue(axon.TargetNeuronId, out float weight))
                    {
                        axon.SynapticWeight = weight;
                    }
                }

                // 跳躍伝導の重みを復元
                var saltatorys = neuron.GetSaltatoryConductionAxons().ToList();
                for (int i = 0; i < saltatorys.Count; i++)
                {
                    var saltatory = saltatorys[i];
                    if (savedState.SaltatoryConductionWeights.TryGetValue(saltatory.TargetNeuronId, out float strength))
                    {
                        saltatory.ConductionStrength = strength;
                    }
                }

                neuron.ActionPotential = savedState.ActionPotential;
            }
        }
        
        // 脳のメモリ履歴を復元
        if (brainData.BrainMemoryHistory != null && brainData.BrainMemoryHistory.Count > 0)
        {
            foreach (var snapshot in brainData.BrainMemoryHistory)
            {
                network.brainMemory.Enqueue(snapshot);
            }
            network.globalTimestamp = brainData.BrainMemoryHistory.Count;
        }

        Console.WriteLine($"[Load] 脳ネットワークを読み込みました: {filePath}");
        Console.WriteLine($"[Load] ニューロン数: {brainData.Neurons.Count}");
        Console.WriteLine($"[Load] 脳メモリスナップショット数: {brainData.BrainMemoryHistory?.Count ?? 0}");
        
        return network;
    }
}

/// <summary>
/// 脳ネットワーク全体のシリアライゼーションデータ
/// </summary>
public class BrainNetworkData
{
    public int InputLayerSize { get; set; }
    public int HiddenLayerSize { get; set; }
    public int OutputLayerSize { get; set; }
    public long Timestamp { get; set; }
    public List<NeuronState> Neurons { get; set; } = new();
    public List<BrainMemorySnapshot> BrainMemoryHistory { get; set; } = new();
}
