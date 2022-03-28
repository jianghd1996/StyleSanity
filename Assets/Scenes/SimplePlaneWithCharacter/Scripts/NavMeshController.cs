using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class NavMeshController : MonoBehaviour
{
    public GameObject positionList;

    private Transform[] transforms;

    private Transform target;

    private NavMeshAgent navMeshAgent;

    void Start() {
        navMeshAgent = GetComponent<NavMeshAgent>();
        transforms = positionList.GetComponentsInChildren<Transform>();
        target = getRandomPosFromList();
    }

    void Update() {
        if ((transform.position-target.position).magnitude < 0.5f) 
          target = getRandomPosFromList();
        //target = getRandomPosFromList();
        navMeshAgent.destination = target.position;
    }

    Transform getRandomPosFromList() {
        return transforms[Random.Range(0, transforms.Length-1)];
    }
}
