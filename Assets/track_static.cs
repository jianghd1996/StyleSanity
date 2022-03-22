using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class track_static : MonoBehaviour
{
    // Start is called before the first frame update
    public Transform target;
    void Start()
    {
        transform.position = target.transform.position + new Vector3(0f, 8f, -5f);
        transform.LookAt(target);
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 p = target.transform.position;
        p.x = 0;
        transform.position = p + new Vector3(0f, 8f, -5f);
    }
}

