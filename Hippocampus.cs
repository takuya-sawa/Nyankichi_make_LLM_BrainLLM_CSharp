using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.IO;

namespace BrainLLM;

/// <summary>
/// 海馬 - アクセス経路の短期記憶と索引管理
/// 生物学的には: 短期記憶の保持、エピソード記憶、空間記憶
/// </summary>
public class Hippocampus
{
    // アクセス経路の履歴（時系列）
    private Queue<AccessPathway> accessHistory = new();
    private const int MaxAccessHistoryCapacity = 100;
    
    // エピソード記憶（時刻とアクセスパターンの対応）
    private Dictionary<long, EpisodicMemory> episodicMemories = new();
    
    // 場所細胞的な空間マップ（どのニューロンIDがどの領域に対応するか）
    private Dictionary<int, SpatialLocation> spatialMap = new();
    
    // 頻繁にアクセスされる経路（長期増強された経路）
    private Dictionary<string, PathwayStrength> frequentPathways = new();
    
    private long currentTimestamp = 0;
    
    // ランダム性（創造性・探索）
    private Random random = new Random();
    private float explorationRate = 0.15f;  // 探索率（15%の確率で新しい経路を試す）
    private float noiseLevel = 0.05f;       // ノイズレベル（記憶の曖昧さ）
    private float forgettingRate = 0.02f;   // 忘却率（2%の記憶が毎回の想起で減衰）
    private int forgettingThreshold = 50;   // 忘却閾値（この時間以上アクセスされないと忘却候補）
    
    public Hippocampus(float explorationRate = 0.15f, float noiseLevel = 0.05f, float forgettingRate = 0.02f)
    {
        this.explorationRate = explorationRate;
        this.noiseLevel = noiseLevel;
        this.forgettingRate = forgettingRate;
        Console.WriteLine($"[Hippocampus] 海馬を初期化しました - アクセス経路記憶システム起動");
        Console.WriteLine($"[Hippocampus] 探索率: {explorationRate:P0}, ノイズレベル: {noiseLevel:F3}, 忘却率: {forgettingRate:P1}");
    }
    
    /// <summary>
    /// 探索率を設定（0.0 = 利用のみ、1.0 = 完全ランダム）
    /// </summary>
    public void SetExplorationRate(float rate)
    {
        explorationRate = Math.Max(0.0f, Math.Min(1.0f, rate));
        Console.WriteLine($"[Hippocampus] 探索率を変更: {explorationRate:P0}");
    }

    /// <summary>
    /// ニューロン間のアクセスを記録
    /// </summary>
    public void RecordAccess(int sourceNeuronId, int targetNeuronId, float signalStrength, string context = "")
    {
        var pathway = new AccessPathway
        {
            SourceNeuronId = sourceNeuronId,
            TargetNeuronId = targetNeuronId,
            SignalStrength = signalStrength,
            Timestamp = currentTimestamp++,
            Context = context
        };
        
        accessHistory.Enqueue(pathway);
        
        // 容量超過時は古い記憶を削除（短期記憶の制限）
        if (accessHistory.Count > MaxAccessHistoryCapacity)
        {
            accessHistory.Dequeue();
        }
        
        // 自動忘却（時々古い記憶を削除）
        if (currentTimestamp % 10 == 0)  // 10アクセスごとに忘却処理
        {
            ForgetOldMemories();
        }
        
        // 頻繁な経路を強化（長期増強 LTP）
        string pathwayKey = $"{sourceNeuronId}->{targetNeuronId}";
        if (frequentPathways.ContainsKey(pathwayKey))
        {
            frequentPathways[pathwayKey].Strength += 0.1f;
            frequentPathways[pathwayKey].AccessCount++;
            frequentPathways[pathwayKey].LastAccessTime = currentTimestamp;
        }
        else
        {
            frequentPathways[pathwayKey] = new PathwayStrength
            {
                SourceId = sourceNeuronId,
                TargetId = targetNeuronId,
                Strength = signalStrength,
                AccessCount = 1,
                FirstAccess = DateTime.UtcNow,
                LastAccessTime = currentTimestamp
            };
        }
    }

    /// <summary>
    /// 脳領域間のアクセスを記録
    /// </summary>
    public void RecordRegionAccess(string sourceRegion, string targetRegion, float[] activationPattern)
    {
        var pathway = new AccessPathway
        {
            SourceNeuronId = -1,  // 領域レベルは特殊ID
            TargetNeuronId = -1,
            SignalStrength = activationPattern.Average(),
            Timestamp = currentTimestamp++,
            Context = $"{sourceRegion} → {targetRegion}"
        };
        
        accessHistory.Enqueue(pathway);
        
        if (accessHistory.Count > MaxAccessHistoryCapacity)
        {
            accessHistory.Dequeue();
        }
    }

    /// <summary>
    /// エピソード記憶を保存（時刻と状態の対応）
    /// </summary>
    public void SaveEpisode(string eventName, Dictionary<int, float> neuronActivations, string context = "")
    {
        var episode = new EpisodicMemory
        {
            EventName = eventName,
            Timestamp = currentTimestamp++,
            NeuronActivations = new Dictionary<int, float>(neuronActivations),
            Context = context,
            RecordedTime = DateTime.UtcNow,
            LastAccessTime = currentTimestamp,
            Importance = 0.5f  // 初期重要度
        };
        
        episodicMemories[episode.Timestamp] = episode;
        
        Console.WriteLine($"[Hippocampus] エピソード記憶保存: '{eventName}' (t={episode.Timestamp})");
    }

    /// <summary>
    /// 過去のアクセス経路を取得（パターン補完 + ランダム探索）
    /// </summary>
    public List<AccessPathway> RecallAccessPattern(int neuronId, int recentSteps = 10)
    {
        // 探索 vs 利用のトレードオフ
        if (random.NextDouble() < explorationRate)
        {
            // 探索: ランダムに新しい経路を提案
            Console.WriteLine($"[Hippocampus] 探索モード: ニューロン{neuronId}の新しい経路を探索中...");
            return GenerateRandomPathways(neuronId, recentSteps);
        }
        
        // 利用: 既知の経路を想起（ノイズ付き）
        var patterns = accessHistory
            .Reverse()
            .Take(recentSteps)
            .Where(p => p.SourceNeuronId == neuronId || p.TargetNeuronId == neuronId)
            .Select(p => AddNoiseToPathway(p))  // ノイズ注入
            .ToList();
        
        return patterns;
    }
    
    /// <summary>
    /// ランダムな経路を生成（創造的思考）
    /// </summary>
    private List<AccessPathway> GenerateRandomPathways(int neuronId, int count)
    {
        var randomPaths = new List<AccessPathway>();
        
        for (int i = 0; i < count; i++)
        {
            var pathway = new AccessPathway
            {
                SourceNeuronId = neuronId,
                TargetNeuronId = random.Next(0, 1000),  // ランダムなターゲット
                SignalStrength = (float)random.NextDouble(),
                Timestamp = currentTimestamp++,
                Context = "Exploratory_Random"
            };
            randomPaths.Add(pathway);
        }
        
        return randomPaths;
    }
    
    /// <summary>
    /// 経路にノイズを追加（記憶の曖昧さ）
    /// </summary>
    private AccessPathway AddNoiseToPathway(AccessPathway pathway)
    {
        float noise = (float)(random.NextDouble() - 0.5) * noiseLevel * 2;
        
        return new AccessPathway
        {
            SourceNeuronId = pathway.SourceNeuronId,
            TargetNeuronId = pathway.TargetNeuronId,
            SignalStrength = Math.Max(0, Math.Min(1, pathway.SignalStrength + noise)),
            Timestamp = pathway.Timestamp,
            Context = pathway.Context + "_Noisy"
        };
    }

    /// <summary>
    /// 頻繁にアクセスされる経路を取得（長期増強された経路）
    /// </summary>
    public List<PathwayStrength> GetFrequentPathways(int minAccessCount = 5)
    {
        return frequentPathways.Values
            .Where(p => p.AccessCount >= minAccessCount)
            .OrderByDescending(p => p.Strength)
            .ToList();
    }

    /// <summary>
    /// エピソード記憶を検索（創造的想起）
    /// </summary>
    public EpisodicMemory? RecallEpisode(string eventName)
    {
        // 探索モード: ランダムに異なるエピソードを返す
        if (random.NextDouble() < explorationRate && episodicMemories.Count > 0)
        {
            var randomIndex = random.Next(episodicMemories.Count);
            var randomEpisode = episodicMemories.Values.ElementAt(randomIndex);
            randomEpisode.LastAccessTime = currentTimestamp;
            randomEpisode.Importance = Math.Min(1.0f, randomEpisode.Importance + 0.1f);  // アクセスで重要度上昇
            Console.WriteLine($"[Hippocampus] 創造的想起: '{randomEpisode.EventName}' (ランダム)");
            return randomEpisode;
        }
        
        // 通常の想起
        var recalledEpisode = episodicMemories.Values
            .FirstOrDefault(e => e.EventName.Contains(eventName, StringComparison.OrdinalIgnoreCase));
        
        if (recalledEpisode != null)
        {
            recalledEpisode.LastAccessTime = currentTimestamp;
            recalledEpisode.Importance = Math.Min(1.0f, recalledEpisode.Importance + 0.1f);
        }
        
        return recalledEpisode;
    }
    
    /// <summary>
    /// 創造的なエピソード記憶の組み合わせ（新しいアイデア生成）
    /// </summary>
    public EpisodicMemory? CreateNovelEpisode(string baseName)
    {
        if (episodicMemories.Count < 2) return null;
        
        // 2つのランダムなエピソードを組み合わせる
        var episode1 = episodicMemories.Values.ElementAt(random.Next(episodicMemories.Count));
        var episode2 = episodicMemories.Values.ElementAt(random.Next(episodicMemories.Count));
        
        var novelActivations = new Dictionary<int, float>();
        
        // 2つのエピソードの活性化パターンを混合
        foreach (var (neuronId, activation) in episode1.NeuronActivations)
        {
            novelActivations[neuronId] = activation * 0.5f;  // 50%
        }
        
        foreach (var (neuronId, activation) in episode2.NeuronActivations)
        {
            if (novelActivations.ContainsKey(neuronId))
                novelActivations[neuronId] += activation * 0.5f;  // もう50%
            else
                novelActivations[neuronId] = activation * 0.5f;
        }
        
        var novelEpisode = new EpisodicMemory
        {
            EventName = $"Novel_{baseName}_{episode1.EventName}+{episode2.EventName}",
            Timestamp = currentTimestamp++,
            NeuronActivations = novelActivations,
            Context = "Creative_Combination",
            RecordedTime = DateTime.UtcNow
        };
        
        Console.WriteLine($"[Hippocampus] 新しいエピソード創造: {novelEpisode.EventName}");
        return novelEpisode;
    }

    /// <summary>
    /// 最近のエピソードを取得
    /// </summary>
    public List<EpisodicMemory> GetRecentEpisodes(int count = 10)
    {
        return episodicMemories.Values
            .OrderByDescending(e => e.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// 空間マップに位置情報を登録（場所細胞）
    /// </summary>
    public void RegisterSpatialLocation(int neuronId, string region, float[] coordinates)
    {
        spatialMap[neuronId] = new SpatialLocation
        {
            NeuronId = neuronId,
            Region = region,
            Coordinates = coordinates,
            RegisteredTime = DateTime.UtcNow
        };
    }

    /// <summary>
    /// 空間的に近いニューロンを検索（探索的ジャンプ付き）
    /// </summary>
    public List<int> FindNearbyNeurons(int referenceNeuronId, float radius = 1.0f)
    {
        if (!spatialMap.ContainsKey(referenceNeuronId))
            return new List<int>();
        
        var refLocation = spatialMap[referenceNeuronId];
        var nearbyNeurons = new List<int>();
        
        foreach (var (neuronId, location) in spatialMap)
        {
            if (neuronId == referenceNeuronId) continue;
            
            float distance = EuclideanDistance(refLocation.Coordinates, location.Coordinates);
            
            // 通常の近傍探索
            if (distance <= radius)
            {
                nearbyNeurons.Add(neuronId);
            }
            // ランダムジャンプ（遠くのニューロンにも接続可能）
            else if (random.NextDouble() < explorationRate * 0.1f)  // 1.5%の確率
            {
                nearbyNeurons.Add(neuronId);
                Console.WriteLine($"[Hippocampus] 探索的ジャンプ: {referenceNeuronId} → {neuronId} (距離={distance:F2})");
            }
        }
        
        return nearbyNeurons;
    }

    /// <summary>
    /// 古い記憶を忘却（時間ベースの減衰）
    /// </summary>
    public void ForgetOldMemories()
    {
        int forgottenPathways = 0;
        int forgottenEpisodes = 0;
        
        // 1. 弱い経路を忘却（強度が減衰して閾値以下になったもの）
        var pathwaysToForget = new List<string>();
        foreach (var kvp in frequentPathways)
        {
            var pathway = kvp.Value;
            long timeSinceAccess = currentTimestamp - pathway.LastAccessTime;
            
            // 時間経過による減衰
            if (timeSinceAccess > forgettingThreshold)
            {
                pathway.Strength *= (1.0f - forgettingRate);
                
                // 強度が0.1以下になったら忘却
                if (pathway.Strength < 0.1f)
                {
                    pathwaysToForget.Add(kvp.Key);
                }
            }
        }
        
        foreach (var key in pathwaysToForget)
        {
            frequentPathways.Remove(key);
            forgottenPathways++;
        }
        
        // 2. 古いエピソード記憶を忘却（最後にアクセスされてから時間が経過したもの）
        var episodesToForget = new List<long>();
        foreach (var kvp in episodicMemories)
        {
            var episode = kvp.Value;
            long timeSinceAccess = currentTimestamp - episode.LastAccessTime;
            
            // アクセスされない記憶は忘れる
            if (timeSinceAccess > forgettingThreshold * 2)  // エピソードは経路より長く保持
            {
                // 重要度が低い記憶を優先的に忘却
                if (episode.Importance < 0.3f || random.NextDouble() < forgettingRate)
                {
                    episodesToForget.Add(kvp.Key);
                }
            }
        }
        
        foreach (var timestamp in episodesToForget)
        {
            episodicMemories.Remove(timestamp);
            forgottenEpisodes++;
        }
        
        if (forgottenPathways > 0 || forgottenEpisodes > 0)
        {
            Console.WriteLine($"[Hippocampus] 忘却処理: 経路 {forgottenPathways}個, エピソード {forgottenEpisodes}個");
        }
    }
    
    /// <summary>
    /// 記憶の統合（Consolidation）- 短期記憶を長期記憶へ
    /// </summary>
    public ConsolidatedMemory ConsolidateMemory()
    {
        var consolidated = new ConsolidatedMemory
        {
            TotalAccessPaths = accessHistory.Count,
            TotalEpisodes = episodicMemories.Count,
            StrongPathways = GetFrequentPathways(minAccessCount: 3),
            ConsolidatedAt = DateTime.UtcNow
        };
        
        Console.WriteLine($"[Hippocampus] 記憶統合完了:");
        Console.WriteLine($"  - アクセス経路: {consolidated.TotalAccessPaths}");
        Console.WriteLine($"  - エピソード記憶: {consolidated.TotalEpisodes}");
        Console.WriteLine($"  - 強化された経路: {consolidated.StrongPathways.Count}");
        
        return consolidated;
    }

    /// <summary>
    /// 海馬の状態を保存
    /// </summary>
    public void Save(string filePath)
    {
        var data = new HippocampusData
        {
            AccessHistory = accessHistory.ToList(),
            EpisodicMemories = episodicMemories.Values.ToList(),
            FrequentPathways = frequentPathways.Values.ToList(),
            SpatialMap = spatialMap.Values.ToList(),
            CurrentTimestamp = currentTimestamp
        };
        
        var options = new JsonSerializerOptions { WriteIndented = true };
        string json = JsonSerializer.Serialize(data, options);
        File.WriteAllText(filePath, json);
        
        Console.WriteLine($"[Hippocampus] 海馬の状態を保存: {filePath}");
    }

    /// <summary>
    /// 海馬の状態を読み込み
    /// </summary>
    public static Hippocampus Load(string filePath)
    {
        string json = File.ReadAllText(filePath);
        var data = JsonSerializer.Deserialize<HippocampusData>(json)
            ?? throw new InvalidOperationException("Failed to load hippocampus data");
        
        var hippocampus = new Hippocampus();
        hippocampus.accessHistory = new Queue<AccessPathway>(data.AccessHistory);
        hippocampus.episodicMemories = data.EpisodicMemories.ToDictionary(e => e.Timestamp, e => e);
        hippocampus.frequentPathways = data.FrequentPathways.ToDictionary(
            p => $"{p.SourceId}->{p.TargetId}", 
            p => p
        );
        hippocampus.spatialMap = data.SpatialMap.ToDictionary(s => s.NeuronId, s => s);
        hippocampus.currentTimestamp = data.CurrentTimestamp;
        
        Console.WriteLine($"[Hippocampus] 海馬の状態を復元: {filePath}");
        Console.WriteLine($"  - アクセス履歴: {hippocampus.accessHistory.Count}");
        Console.WriteLine($"  - エピソード: {hippocampus.episodicMemories.Count}");
        
        return hippocampus;
    }

    /// <summary>
    /// 海馬の統計情報を表示
    /// </summary>
    public void PrintStats()
    {
        Console.WriteLine("\n[Hippocampus Statistics]");
        Console.WriteLine($"  アクセス履歴: {accessHistory.Count}/{MaxAccessHistoryCapacity}");
        Console.WriteLine($"  エピソード記憶: {episodicMemories.Count}");
        Console.WriteLine($"  頻繁な経路: {frequentPathways.Count}");
        Console.WriteLine($"  空間マップ: {spatialMap.Count} neurons");
        
        if (frequentPathways.Count > 0)
        {
            Console.WriteLine("\n  [最も強化された経路 Top 5]:");
            foreach (var pathway in GetFrequentPathways(1).Take(5))
            {
                Console.WriteLine($"    {pathway.SourceId} → {pathway.TargetId}: " +
                    $"強度={pathway.Strength:F3}, アクセス={pathway.AccessCount}回");
            }
        }
    }

    private float EuclideanDistance(float[] a, float[] b)
    {
        if (a.Length != b.Length) return float.MaxValue;
        
        float sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (float)Math.Sqrt(sum);
    }
}

/// <summary>
/// アクセス経路（シナプス伝達の記録）
/// </summary>
public class AccessPathway
{
    public int SourceNeuronId { get; set; }
    public int TargetNeuronId { get; set; }
    public float SignalStrength { get; set; }
    public long Timestamp { get; set; }
    public string Context { get; set; } = "";
}

/// <summary>
/// エピソード記憶（時刻と状態の対応）
/// </summary>
public class EpisodicMemory
{
    public string EventName { get; set; } = "";
    public long Timestamp { get; set; }
    public Dictionary<int, float> NeuronActivations { get; set; } = new();
    public string Context { get; set; } = "";
    public DateTime RecordedTime { get; set; }
    public long LastAccessTime { get; set; }  // 最後にアクセスされた時刻
    public float Importance { get; set; } = 0.5f;  // 重要度（0.0～1.0）
}

/// <summary>
/// 経路の強度（長期増強された接続）
/// </summary>
public class PathwayStrength
{
    public int SourceId { get; set; }
    public int TargetId { get; set; }
    public float Strength { get; set; }
    public int AccessCount { get; set; }
    public DateTime FirstAccess { get; set; }
    public long LastAccessTime { get; set; }  // 最後にアクセスされた時刻
}

/// <summary>
/// 空間的位置情報（場所細胞）
/// </summary>
public class SpatialLocation
{
    public int NeuronId { get; set; }
    public string Region { get; set; } = "";
    public float[] Coordinates { get; set; } = Array.Empty<float>();
    public DateTime RegisteredTime { get; set; }
}

/// <summary>
/// 統合された記憶
/// </summary>
public class ConsolidatedMemory
{
    public int TotalAccessPaths { get; set; }
    public int TotalEpisodes { get; set; }
    public List<PathwayStrength> StrongPathways { get; set; } = new();
    public DateTime ConsolidatedAt { get; set; }
}

/// <summary>
/// 海馬のシリアライゼーションデータ
/// </summary>
public class HippocampusData
{
    public List<AccessPathway> AccessHistory { get; set; } = new();
    public List<EpisodicMemory> EpisodicMemories { get; set; } = new();
    public List<PathwayStrength> FrequentPathways { get; set; } = new();
    public List<SpatialLocation> SpatialMap { get; set; } = new();
    public long CurrentTimestamp { get; set; }
}
