using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UnevenGround : MonoBehaviour {

    [SerializeField] private MeshFilter filter;
    [SerializeField] private MeshRenderer renderer;
    [SerializeField] private MeshCollider collider;

    [Space] [SerializeField] private float radius;
    [SerializeField] private int steps;

    [Space] [SerializeField] private float maxHeight;
    [SerializeField] private float scale;

    public void Generate(float steepness) {

        var verts = new List<Vector3>();
        var tris = new List<int>();
        var uvs = new List<Vector2>();

        var ssize = 2 * radius / steps;

        for (int x = 0; x <= steps; x++) {
            for (int y = 0; y <= steps; y++) {

                var vert = new Vector3(x * ssize - radius, 0, y * ssize - radius);
                var dist = vert.magnitude;

                var nois = -1 * Mathf.PerlinNoise((vert.x + radius) * scale * 1.1f, (vert.z + radius) * scale * 1.1f) * maxHeight * steepness + 0.5f;
                vert.y = Mathf.Lerp(0.5f, nois, Mathf.Pow(Mathf.Clamp01(dist / 13), 2));

                if (x == 0 || y == 0 || x == steps || y == steps) vert.y = 0.5f;
                
                verts.Add(vert);
                uvs.Add(new Vector2(x, y) / steps * radius);
                if (x == 0 || y == 0) continue;

                print(uvs[uvs.Count - 1]);

                var index = verts.Count - 1;

                tris.Add(index);
                tris.Add(index - 1);
                tris.Add(index - steps - 2);

                tris.Add(index);
                tris.Add(index - steps - 2);
                tris.Add(index - steps - 1);
            }
        }

        var mesh = new Mesh();
        mesh.SetVertices(verts);
        mesh.SetTriangles(tris, 0);
        mesh.SetUVs(0, uvs);

        filter.sharedMesh = mesh;
        collider.sharedMesh = mesh;
    }

    private void OnDrawGizmosSelected() {

        Gizmos.color = Color.yellow;
        Gizmos.DrawWireCube(transform.position, new Vector3(radius * 2, 2, radius * 2));
    }
}
