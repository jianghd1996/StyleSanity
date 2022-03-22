using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class track : MonoBehaviour
{
    public Transform target;
    Vector3 T;
    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        transform.position = target.transform.position + (target.transform.position-navigate_3D.target).normalized * 3 + new Vector3(1f, 2f, 0);
        transform.LookAt(target);
    }
}
