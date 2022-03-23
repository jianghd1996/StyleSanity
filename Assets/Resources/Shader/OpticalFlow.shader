// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Hidden/OpticalFlow"
{
	Properties
	{
		_Sensitivity("Sensitivity", Float) = 1
	}
	SubShader
	{
		// No culling or depth
		Cull Off ZWrite Off ZTest Always

		Pass
		{
			CGPROGRAM
			#pragma target 5.0
			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"

 
			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
			};

			float4 _CameraMotionVectorsTexture_ST;
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _CameraMotionVectorsTexture);
				return o;
			}
			
			sampler2D _CameraMotionVectorsTexture;

            float3 Hue(float H)
			{
			    float R = abs(H * 6 - 3) - 1;
			    float G = 2 - abs(H * 6 - 2);
			    float B = 2 - abs(H * 6 - 4);
			    return saturate(float3(R,G,B));
			}

			float3 HSVtoRGB1(float3 HSV)
			{
			    return float3(((Hue(HSV.x) - 1) * HSV.y + 1) * HSV.z);
			}

			float3 HSVtoRGB2 (float3 HSV) {
				float hue = HSV.x;
				float sat = HSV.y;
				float val = HSV.z;

				float x = 0, y = 0, z = 0;
        // normalize + amplication
				if (hue == 1) hue = 0;
				else hue *= 6;

				int i = floor(hue);
				float f = hue-i;
				float p = val*(1-sat);
				float q = val*(1-(sat*f));
				float t = val*(1-(sat*(1-f)));

				switch (i) 
				{
					case 0: x = val; y = t; z = p; break;
					case 1: x = q; y = val; z = p; break;
					case 2: x = p; y = val; z = t; break;
					case 3: x = p; y = q; z = val; break;
					case 4: x = t; y = p; z = val; break;
					case 5: x = val; y = p; z = q; break;
				}

				return float3(x,y,z);
			}

			float _Sensitivity;
			// RWStructuredBuffer<float2> _Motions : register(u1);
			// uniform int _Width;
			// uniform int _Height;

			float3 MotionVectorsToOpticalFlow(float2 motion)
			{
				// Currently is based on HSV encoding from:
				//			"Optical Flow in a Smart Sensor Based on Hybrid Analog-Digital Architecture" by P. Guzman et al
				//			http://www.mdpi.com/1424-8220/10/4/2975

				// Analogous to http://docs.opencv.org/trunk/d7/d8b/tutorial_py_lucas_kanade.html
				// but might need to swap or rotate axis!

				// @TODO: support other HSV encodings (using lookup texture)
				// https://www.microsoft.com/en-us/research/wp-content/uploads/2007/10/ofdatabase_iccv_07.pdf
				// https://people.csail.mit.edu/celiu/SIFTflow/
				// some MATLAB code: https://github.com/suhangpro/epicflow/blob/master/utils/flow-code-matlab/computeColor.m

				float angle = atan2(-motion.y, -motion.x); // radian between -pi to pi?

				float hue = angle / (UNITY_PI * 2.0) + 0.5;		// convert motion angle to Hue
				float value = length(motion) * _Sensitivity;  	// convert motion strength to Value
    			return HSVtoRGB2(float3(hue, 1, value));		// HSV -> RGB
			}

			fixed4 frag (v2f i) : SV_Target
			{
				float2 motion = tex2D(_CameraMotionVectorsTexture, i.uv).rg;
				
				float3 rgb = MotionVectorsToOpticalFlow(motion);

				// int x = i.uv.x * _Width;
				// int y = i.uv.y * _Height;
				// int id = x + _Width * y;
				// _Motions[id] = motion;

				return float4(rgb, 1);
				//return float4(motion.x, motion.y, 0, 1)*_Sensitivity;
			}
			ENDCG
		}
	}
}
