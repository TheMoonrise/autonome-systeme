using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class WorldConfig : MonoBehaviour {

    [SerializeField] private PhysicMaterial groundPhysicsMat;
    [SerializeField] private Material groundMat;
    [SerializeField] private Color defaultColor, iceColor;

    private float current_slipperiness = -1;
    private float current_steepness = -1;

    private void Awake() {

        UpdateSteepness(0);
        UpdateSlipperiness(0);
    }

    void Update() {

        var eparams = Academy.Instance.EnvironmentParameters;

        var slipperiness = eparams.GetWithDefault("slipperiness", 0);
        UpdateSlipperiness(slipperiness);

        var steepness = eparams.GetWithDefault("steepness", 0);
        UpdateSteepness(steepness);
    }

    private void UpdateSlipperiness(float slipperiness) {

        if (current_slipperiness == slipperiness) return;
        current_slipperiness = slipperiness;

        var color = Color.Lerp(defaultColor, iceColor, slipperiness);
        groundMat.SetColor("_Color", color);

        var friction = Mathf.Lerp(.6f, 0f, slipperiness);
        groundPhysicsMat.dynamicFriction = friction;
        groundPhysicsMat.staticFriction = friction;
    }

    private void UpdateSteepness(float steepness) {

        if (current_steepness == steepness) return;
        current_steepness = steepness;

        foreach (var g in FindObjectsOfType<UnevenGround>()) g.Generate(steepness);
    }
}
