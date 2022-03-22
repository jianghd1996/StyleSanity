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

using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using UnityEngine;
using UnityEngine.Playables;
using UnityEngine.SceneManagement;
using UnityEditor;
using System.IO;
using System.Threading;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class CameraAgentForrest : Agent
{

    private Camera main;

    public Transform SceneCenter, SceneCorner;

    private RenderTexture renderTexture;
    private Texture2D screenShot;
    public float modelSize;

    private float theta, deltaSpeed, AngleSpeed, MaxSpeed, distance, Maxdistance;
    private const float speedNorm = 10.0f;
    private float[] occupancy_map;
    private int Maxdetectors;

    private float rewardCollision;

    public int recordedStep=0;
    EnvironmentParameters resetParams;

    public float timeScaleValue=1.0f;
    
    public float RaycastDist = 5.0f;

    Vector3 direction, Center;

    public Color color;
    public float Width;
    public Material Mat;
    LineRenderer L;

    LineRenderer Add_line(Vector3 start, Vector3 end, Material M) {
        GameObject child = new GameObject();
        LineRenderer line = child.AddComponent<LineRenderer>();

        line.enabled = true;
        line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        line.receiveShadows = false;
        line.material = M;
        line.startColor = color;
        line.endColor = color;
        line.startWidth = Width;
        line.endWidth = Width;
        line.positionCount =3;

        line.SetPosition(0, start);
        line.SetPosition(1, end);
        line.SetPosition(2, end);

        return line;
    }

    public override void Initialize()
    {
        // Initialize env
        resetParams = Academy.Instance.EnvironmentParameters;
        main = gameObject.GetComponent<Camera>();

        // Application.targetFrameRate = 30;
        // renderTexture = new RenderTexture(Screen.width, Screen.height, 24);
        // screenShot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);

        deltaSpeed = 0.1f / speedNorm;
        MaxSpeed = 0.8f / speedNorm;
        Maxdetectors = 60;
        Maxdistance = (SceneCorner.transform.position-SceneCenter.transform.position).magnitude * 0.75f;
        Center = SceneCenter.transform.position + new Vector3(0, 5, 0);

        Time.timeScale=timeScaleValue;
        Application.runInBackground=true;
        L = Add_line(Center, SceneCorner.transform.position, Mat);

        // Display.displays[0].Deactivate();
    }

    public override void OnEpisodeBegin()
    {
        // Initialize the camera and cube position
        this.initializeScene();
        recordedStep=0;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(AngleSpeed);
        sensor.AddObservation(Vector3.Dot(direction, Vector3.Normalize(Center-transform.position)));
        sensor.AddObservation(distance);
        sensor.AddObservation(occupancy_map);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        recordedStep+=1;

        // Update camera angle according to the received action
        // Discrete agent
        int action = actionBuffers.DiscreteActions[0];
        rewardCollision = 1;

        if (action == 0) AngleSpeed += deltaSpeed;
        else if (action == 1) AngleSpeed -= deltaSpeed;

        if (Mathf.Abs(AngleSpeed) > MaxSpeed) {
            rewardCollision -= 0.5f;
            if (action == 0) AngleSpeed -= deltaSpeed;
            else if (action == 1) AngleSpeed += deltaSpeed;
        }
        rewardCollision -= Mathf.Abs(AngleSpeed) * speedNorm;

        theta += AngleSpeed;

        direction = new Vector3(Mathf.Cos(theta), 0, Mathf.Sin(theta));
        transform.position += direction / speedNorm;
        transform.LookAt(transform.position+direction);

        distance = (transform.position - Center).magnitude / Maxdistance;



        if (distance > 1) {
            rewardCollision -= 0.3f;
            transform.position -= direction / speedNorm;
        }

        get_occupancy_map();

        for (int i = 0; i < Maxdetectors; ++i)
            if (occupancy_map[i] != 0 && occupancy_map[i] < 0.5f) {
              rewardCollision -= 1f;
              break;
            }

        // Debug.Log(rewardCollision.ToString()+" "+(AngleSpeed*speedNorm).ToString()+" "+distance.ToString());

        SetReward(rewardCollision);
    }

    private void get_occupancy_map() {
        occupancy_map = new float[Maxdetectors];
        float min_theta = Mathf.PI * 2 / Maxdetectors;

        for (int i = 0; i < Maxdetectors; ++i) {
          Vector3 forward = new Vector3(Mathf.Cos(theta+min_theta*i), 0, Mathf.Sin(theta+min_theta*i)) * 3;
          Vector3 detect_point = transform.position + forward;
          Ray ray = new Ray(transform.position, forward);
          bool isCollider = Physics.Raycast(ray, out RaycastHit hit, forward.magnitude);
          if (isCollider) {
              occupancy_map[i] = (transform.position-hit.point).magnitude;
              Debug.DrawLine(transform.position, detect_point, Color.red);
          }
          else {
              occupancy_map[i] = 0;
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

    private void initializeScene()
    {
        theta = Random.Range(0, 1f) * Mathf.PI * 2;

        transform.position = Center + new Vector3(Mathf.Cos(theta), 0, Mathf.Sin(theta)) * Random.Range(0, 0.8f) * Maxdistance;
        
        direction = new Vector3(Mathf.Cos(theta), 0, Mathf.Sin(theta));
        transform.LookAt(transform.position+direction);

        distance = (transform.position - Center).magnitude / Maxdistance;
        AngleSpeed = 0.1f / speedNorm;
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