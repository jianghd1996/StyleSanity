/*
Requirements:

    - Install `ml-agent` through the Package Manager

    - Add this script to the agent (camera)
        + Set the target.s

    - Add a `Decision Requester` to the agent
        + Set the decision period

    - Add a `Behavior Parameter`
        + Set up the name: `CameraControl`
        + Set the observation space:
            * Space size = x
            * Stacked vector = 1
        + Set the action space:
            * Continuous actions = 0
            * Discrete actions = 2 (L, R)
*/


using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

public class corridor : Agent
{

    private Camera main;

    private float theta, deltaSpeed, distance;
    private const float speedNorm = 50.0f;
    private float[] occupancy_map;
    private int Maxdetectors;
    private float detect_dist;

    private float rewardCollision;

    public int recordedStep=0;
    EnvironmentParameters resetParams;

    public float timeScaleValue=1.0f;

    private int action, a, r_action;

    Vector3 direction;

    public Material Mat;
    public GameObject PrefabModel, cylinder;
    private GameObject newModel;
    private Queue<GameObject> Cylinder;
    public float ModelSize;

    void SetLayerRecursively(GameObject obj, int newLayer)
    {
        if (null == obj)
        {
            return;
        }
       
        obj.layer = newLayer;
       
        foreach (Transform child in obj.transform)
        {
            if (null == child)
            {
                continue;
            }
            SetLayerRecursively(child.gameObject, newLayer);
        }
    }

    public override void Initialize()
    {
        // Initialize env
        resetParams = Academy.Instance.EnvironmentParameters;
        main = gameObject.GetComponent<Camera>();

        deltaSpeed = 0.2f / speedNorm;
        Maxdetectors = 60;
        detect_dist = 3f;
        Cylinder = new Queue<GameObject>();

        Time.timeScale=timeScaleValue;
        Application.runInBackground=true;

        newModel = Instantiate(PrefabModel, transform.position, transform.rotation);
                newModel.GetComponentsInChildren<Renderer>().ToList().ForEach(M =>
                {
                    M.sharedMaterial = Mat;
                    M.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    M.receiveShadows = false;
                });
                newModel.transform.localScale = Vector3.one * ModelSize;
                SetLayerRecursively(newModel, 1);

    }

    public override void OnEpisodeBegin()
    {
        // Initialize the camera and cube position
        this.initializeScene();
        recordedStep=0;
    }

    public override async void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(r_action);
        sensor.AddObservation(occupancy_map);
    }

    public override async void OnActionReceived(ActionBuffers actionBuffers)
    {
        recordedStep+=1;
        
        // Update camera angle according to the received action
        // Discrete agent
        action = actionBuffers.DiscreteActions[0];
        rewardCollision = 0;

        if (action > 1)
            action = a;
        
        r_action = action;

        if (action == 0)
            direction = new Vector3(-1/speedNorm, 0, 1/speedNorm);
        else
            direction = new Vector3(1/speedNorm, 0, 1/speedNorm);
        
        transform.position += direction;

        GameObject x;

        if (Cylinder.Peek().transform.position.z < transform.position.z-2) {
            float z = Cylinder.Peek().transform.position.z;
            x = Cylinder.Peek();
            Cylinder.Dequeue(); 
            Destroy(x);
            x = Cylinder.Peek();
            Cylinder.Dequeue();
            Destroy(x);
            distance = Mathf.Clamp(distance+Random.Range(-1.0f, 1.0f), 1, 4);
            Cylinder.Enqueue(Instantiate(cylinder, new Vector3(-distance, 0, z+5), transform.rotation));
            Cylinder.Enqueue(Instantiate(cylinder, new Vector3(distance, 0, z+5), transform.rotation));
        }

        newModel.transform.position = transform.position;
        newModel.transform.rotation = transform.rotation;

        get_occupancy_map();

        int flag = 1;

        for (int i = 0; i < Maxdetectors; ++i)
            if (occupancy_map[i] * detect_dist < 0.1f) {
              rewardCollision -= 1f;
              flag = 0;
              Debug.Log("Crash");
              break;
            }
        
        SetReward(rewardCollision);

        if (flag == 0) {
            while (Cylinder.Count > 0) {
                x = Cylinder.Peek();
                Cylinder.Dequeue(); 
                Destroy(x);
            }
            EndEpisode();
        }
    }

    private void get_occupancy_map() {
        occupancy_map = new float[Maxdetectors];
        float min_theta = Mathf.PI * 2 / Maxdetectors;

        for (int i = 0; i < Maxdetectors; ++i) {
          Vector3 forward = new Vector3(Mathf.Cos(theta+min_theta*i), 0, Mathf.Sin(theta+min_theta*i)) * detect_dist;
          Vector3 detect_point = transform.position + forward;
          Ray ray = new Ray(transform.position, forward);
          bool isCollider = Physics.Raycast(ray, out RaycastHit hit, forward.magnitude);
          if (isCollider) {
              occupancy_map[i] = (transform.position-hit.point).magnitude / detect_dist;
              Debug.DrawLine(transform.position, detect_point, Color.red);
          }
          else {
              occupancy_map[i] = 1.0f;
              Debug.DrawLine(transform.position, detect_point, Color.green);
          }
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // // Heuristic method to test the env
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKey("left"))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey("right"))
        {
            discreteActionsOut[0] = 0;
        }
        else discreteActionsOut[0] = 2;
        // var continuousActionsOut = actionsOut.ContinuousActions;
        // continuousActionsOut[0] = Input.GetAxis("Horizontal");
    }

    private async void initializeScene()
    {
        Random.InitState((int)System.DateTime.Now.Ticks);
        a = Random.Range(0, 2);
        action = a;
        r_action = a;

        transform.position = new Vector3(0, 0, 0);
        direction = new Vector3(0, 0, 1);
        transform.LookAt(transform.position+direction);

        Cylinder.Clear();

        distance = 2f + Random.Range(0, 1.0f) * 8f;
        for (int i = 0; i < 5; ++i) {
            Cylinder.Enqueue(Instantiate(cylinder, new Vector3(distance, 0, i-1), transform.rotation));
            Cylinder.Enqueue(Instantiate(cylinder, new Vector3(-distance, 0, i-1), transform.rotation));
            distance = Mathf.Clamp(distance+Random.Range(-1.0f, 1.0f), 1, 4);
        }

        get_occupancy_map();
    }

    /*
    private bool isColliding(Vector3 obstacle)
    {
        Vector3 head = target.transform.position;
        Vector3 hc = obstacle - head;
        Vector3 hd = transform.position - head;
        
        float angle = Vector3.Angle(hc, hd);

        Debug.DrawRay(head, hc);
        Debug.DrawRay(head, hd);

        return angle < 45f;
    }

    private bool isCollidingRC()
    {
        RaycastHit hit;
        // Bit shift the index of the layer (3) to get a bit mask
        int layerObs = 1 << 3;

        Vector3 dirRC = (target.transform.position - transform.position).normalized;
        if (Physics.Raycast(transform.position, dirRC, out hit, RaycastDist, layerObs))
        {
            Debug.DrawRay(transform.position, dirRC * hit.distance, Color.red);
            // Debug.Log("Did Hit");
            return true;
        }
        else
        {
            Debug.DrawRay(transform.position, dirRC * RaycastDist, Color.green);
            // Debug.Log("Did not Hit");
            return false;
        }
    }
    */
}