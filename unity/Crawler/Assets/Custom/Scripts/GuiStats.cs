using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class GuiStats : MonoBehaviour {

    [SerializeField] private CrawlerAgent agent;

    private List<float> rewards = new List<float>();

    private float lastEpisodeReward = 0;
    private float averageReward = 0;

    private float lastReward = 0;
    private float lastStepCount = 0;

    private void OnGUI() {

        var reward = agent.GetCumulativeReward();
        var steps = agent.StepCount;

        if (steps < lastStepCount) {
            lastEpisodeReward = lastReward;
            rewards.Add(lastReward);
            if (rewards.Count > 100) rewards.RemoveAt(0);
            averageReward = rewards.Average();
        }

        lastStepCount = steps;
        lastReward = reward;

        var info = $"Step: {steps:0000}/{agent.MaxStep}\nReward: {reward:0.00}\nLast Reward: {lastEpisodeReward:0.00}\nAverage Reward (100): {averageReward:0.00}";
        GUI.Label(new Rect(10, 10, 300, 100), info);
    }
}
