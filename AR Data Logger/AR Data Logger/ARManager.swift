//
//  ARManager.swift
//  AR Data Logger
//
//  Created by Paul Ruvolo on 6/22/22.
//

import Foundation
import ARKit
import RealityKit
import ARDataLogger
import FirebaseAuth
import FirebaseCore

class ARManager: NSObject, ARSessionDelegate {
    public static var shared = ARManager()
    
    let arView = ARView(frame: .zero)
    var lastFrameLogTime = Date()
    
    private override init() {
        super.init()
        ARDataLogger.ARLogger.shared.dataDir = "visual_alignment_benchmarking"
        ARDataLogger.ARLogger.shared.doAynchronousUploads = false
        FirebaseApp.configure()
        if Auth.auth().currentUser == nil {
            Auth.auth().signInAnonymously() { (authResult, error) in
                print("authResult \(authResult)")
            }
        }
        let configuration = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics = .sceneDepth
        }
        arView.session.delegate = self
        ARDataLogger.ARLogger.shared.startTrial()
        arView.session.run(configuration)
    }
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        ARDataLogger.ARLogger.shared.session(session, didUpdate: frame)
        if -lastFrameLogTime.timeIntervalSinceNow > 2.0 {
            // log frame here
            ARDataLogger.ARLogger.shared.log(frame: frame, withType: "benchmarking", withMeshLoggingBehavior: .none)
            lastFrameLogTime = Date()
        }
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        ARDataLogger.ARLogger.shared.session(session, didAdd: anchors)
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        ARDataLogger.ARLogger.shared.session(session, didUpdate: anchors)
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        ARDataLogger.ARLogger.shared.session(session, didRemove: anchors)
    }
}
