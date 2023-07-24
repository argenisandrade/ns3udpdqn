/*
 * wifi-adhoc-b-udp-cc-dqn.cc
 *
 *  Created on: 16 jul 2023
 *      Author: Argenis Andrade
 */
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/log.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/mobility-model.h"
#include "ns3/olsr-helper.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/flow-monitor.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/event-id.h"
#include "ns3/netanim-module.h"
#include "ns3/wifi-helper.h"
#include "ns3/spectrum-module.h"
#include "ns3/wifi-module.h" 
#include "ns3/core-module.h"
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <cstdlib>  // needed for rand()
#include <ctime>  // needed to properly seed random number generator
#include "ns3/ns3-ai-module.h"

using namespace std;
using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("wifi-adhoc-b-udp-cc-dqn");

struct Env
{
    double node;
    double app;
    double cbr;
    double pdr;
    double cf;
    double bitrate;
    double ql;
}Packed;

struct Act
{
    double pred;
}Packed;

class CC_DQN : public Ns3AIRL<Env, Act>
{
public:
    CC_DQN(uint16_t id);
    int Func(double node, double app, double cbr, double pdr, double cf, double bitrate, double ql);
};

CC_DQN::CC_DQN(uint16_t id) : Ns3AIRL<Env, Act>(id) {
    SetCond(2, 0);      ///< Set the operation lock (even for ns-3 and odd for python).
}

int CC_DQN::Func(double node, double app, double cbr, double pdr, double cf, double bitrate, double ql)
{
    auto env = EnvSetterCond();     ///< Acquire the Env memory for writing
    env->node = node;
    env->app = app;
    env->cbr = cbr;
    env->pdr = pdr;
    env->cf = cf;
    env->bitrate = bitrate;
    env->ql = ql;
    SetCompleted();                 ///< Release the memory and update conters
    NS_LOG_DEBUG ("Ver:" << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_id));
    auto act = ActionGetterCond();  ///< Acquire the Act memory for reading
    double ret = act->pred;
    GetCompleted();                 ///< Release the memory, roll back memory version and update conters
    NS_LOG_DEBUG ("Ver:" << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_id));
    return ret;
}

int memblock_key = 2345;        ///< memory block key, need to keep the same in the python script

double c = 1.0;			//Prediction
CC_DQN cc_dqn(memblock_key);

//===Global variables to generate the Training Data Packets
NetDeviceContainer devicesGlobal;
Ipv4InterfaceContainer interfacesGlobal;
NodeContainer nodesGlobal;

FlowMonitorHelper flowmon;
Ptr<FlowMonitor> monitor;

bool FlowMonitorFlows;
bool FlowMonitorResults;

bool m_enableApp1;
uint32_t m_portApp1;
uint32_t m_packetSizeMeanApp1;
double m_packetIntervalMeanApp1;
bool m_enableApp2;
uint32_t m_portApp2;
uint32_t m_packetSizeMeanApp2;
double m_packetIntervalMeanApp2;
double a = 2; //patrones de activacion
double routeTime = 30.0;
uint32_t numNodes = 16;
uint32_t sinkNode = 0;
uint32_t numSeeds = 4;
uint32_t numRun = 5;
double mode;

//std::string topology = "SquareGrid";
std::string topology = "Mobile";
//std::string topology = "Montreal";


//std::string TrafficPattern = "Random";
std::string TrafficPattern = "Constant";
//std::string TrafficPattern = "RollerCoaster";

double predictInterval = 1.0;

std::ofstream tracesResultsFile; 
//std::ofstream CBRsResultsFile;
std::ofstream PDRsResultsFile;

double measureMean=0.001;
EventId sendEvent;
double m_weight = 0.005;
double m_weightPDR = 0.05;
double m_weightCF = 0.05;
uint64_t m_topDelay = 50000000; // 50 miliseconds
std::map<uint32_t,double>  m_queueEWMA;// Almacenar el valor de queue anterior
std::map<uint32_t,double>  m_rhoEWMA;// Almacenar el valor de rho por interfaz
std::map<std::pair<uint32_t, uint32_t>,uint32_t> m_pktTX; // Storage transmitted packages
std::map<std::pair<uint32_t, uint32_t>,uint32_t> m_pktRX; // Storage received packages
std::map<std::pair<uint32_t, uint32_t>,double> m_nodePDR; // Storage packet delivery ratio of nodes
std::map<std::pair<uint32_t, uint32_t>,double> m_pdrEWMA;

std::map<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>,uint64_t> m_timeTX;
std::map<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>,uint64_t> m_timeRX;
std::map<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>,uint64_t> m_timeRXCF;
std::map<std::pair<uint32_t, uint32_t>,uint32_t> m_pktRXCF; // Storage received packages on time
std::map<std::pair<uint32_t, uint32_t>,double> m_nodeCF; // Storage compliant factor of node
std::map<std::pair<uint32_t, uint32_t>,double> m_cfEWMA;

std::map<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>,uint32_t> m_sizeTX;
std::map<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>,uint32_t> m_sizeRX;
std::map<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>,uint32_t> m_sizeRXCT;
std::map<std::pair<uint32_t, uint32_t>,double> m_BitRateTX;
std::map<std::pair<uint32_t, uint32_t>,double> m_BitRateRX;
std::map<std::pair<uint32_t, uint32_t>,double> m_BitRateRXCT;
//std::map<uint32_t, uint32_t> m_numReps;

std::map<std::pair<uint32_t, uint32_t>,double>  m_predict;
int pktNT = 0;

void CBRSample ( uint32_t node,
	NodeContainer _nodes,
	NetDeviceContainer _devices,
	double _simulationTime)
{
	Ptr<NetDevice> nd = _devices.Get (node);
	Ptr<WifiNetDevice> wnd = nd->GetObject<WifiNetDevice> ();

	Ptr<WifiMac> wifi_mac = wnd->GetMac();
	Ptr<RegularWifiMac> reg_wifi_mac = DynamicCast<RegularWifiMac>(wifi_mac);
	Ptr<WifiMacQueue> wmq=reg_wifi_mac->GetBEQueue()->GetWifiMacQueue();

	int m_currentRho;
	if (wnd->GetMac()->GetWifiPhy()->IsStateIdle()) {
		m_currentRho = 0;
	} else {
		m_currentRho = 1;
	}

	m_rhoEWMA[node] = m_weight * m_currentRho + (1 - m_weight) * m_rhoEWMA[node];

	uint32_t currentQP=wmq->GetNPackets();
	m_queueEWMA[node] = (m_weight * currentQP + (1 - m_weight) * m_queueEWMA[node]);

	/*CBRsResultsFile << Simulator::Now().GetNanoSeconds() << "\t"
			<< node << "\t"
			<< m_rhoEWMA[node]<< "\t"
			<< m_queueEWMA[node] << "\t"
			<< m_currentRho<< "\t"
			<< currentQP << "\t"
			<< numSeeds << "\t"
			<< mode << "\t"
			<< topology <<"\t"
			<< TrafficPattern << "\t"
		    << predictInterval << "\t"
			<< numNodes << "\n";*/

	Ptr<ExponentialRandomVariable> _interval = CreateObject<ExponentialRandomVariable> ();
	double newMeasure = _interval->GetValue(measureMean,0); // The new measure has to be done without memory in order to avoid syncronization

	if (Simulator::Now ().GetSeconds()<_simulationTime){ // Metodo recursivo
		sendEvent=Simulator::Schedule(Seconds(newMeasure), &CBRSample , node, _nodes, _devices, _simulationTime) ;
	}
	else{
		Simulator::Cancel (sendEvent);
	}
}

void ComputePDR_and_CF ( double _simulationTime )
{
	std::map<std::pair<uint32_t, uint32_t>,uint32_t> counterTX;
	std::map<std::pair<uint32_t, uint32_t>,uint32_t> counterRX;
	std::map<std::pair<uint32_t, uint32_t>,uint32_t> counterRXCF;
	std::map<std::pair<uint32_t, uint32_t>,uint32_t> accumTX;
	std::map<std::pair<uint32_t, uint32_t>,uint32_t> accumRX;
	std::map<std::pair<uint32_t, uint32_t>,uint32_t> accumRXCT;

	if (Simulator::Now ().GetSeconds()<_simulationTime){ // Metodo recursivo

		std::cout << "\n"<< Simulator::Now ().GetMilliSeconds() << " | Seed: " << numSeeds << " | Mode: " << mode << " | Topology: " << topology << " | Pattern: " << TrafficPattern<< " | PredInterval: " << predictInterval <<std::endl;

		// Iteration for transmitted packages 
		for (std::map<std::pair<uint32_t, uint32_t>,uint32_t>::const_iterator iterator = m_pktTX.begin();
													iterator != m_pktTX.end(); iterator++)
			{
					std::map<std::pair<uint32_t, uint32_t>, uint32_t>::iterator i = counterTX.find (std::make_pair(iterator->second, iterator->first.second));
					  if (i == counterTX.end ())
						{
							 counterTX[std::make_pair(iterator->second, iterator->first.second)] = 0;
						}
					counterTX[std::make_pair(iterator->second, iterator->first.second)]++;
			}
			
		// Iteration for received packages
		for (std::map<std::pair<uint32_t, uint32_t>,uint32_t>::const_iterator iterator = m_pktRX.begin();
															iterator != m_pktRX.end(); iterator++)
			{
					std::map<std::pair<uint32_t, uint32_t>, uint32_t>::iterator i = counterRX.find (std::make_pair(iterator->second, iterator->first.second));
					  if (i == counterRX.end ())
						{
							 counterRX[std::make_pair(iterator->second, iterator->first.second)] = 0;
						}
					counterRX[std::make_pair(iterator->second, iterator->first.second)]++;
			}

		// Iteration for received packages on time
		for (std::map<std::pair<uint32_t, uint32_t>,uint32_t>::const_iterator iterator = m_pktRXCF.begin();
															iterator != m_pktRXCF.end(); iterator++)
			{
					std::map<std::pair<uint32_t, uint32_t>, uint32_t>::iterator i = counterRXCF.find (std::make_pair(iterator->second, iterator->first.second));
					  if (i == counterRXCF.end ())
						{
							 counterRXCF[std::make_pair(iterator->second, iterator->first.second)] = 0;
						}
					counterRXCF[std::make_pair(iterator->second, iterator->first.second)]++;
			}

		// Iteration for size of transmitted packages
		for (std::map<std::pair< uint32_t, std::pair<uint32_t, uint32_t>>,uint32_t>::const_iterator iterator = m_sizeTX.begin();
															iterator != m_sizeTX.end(); iterator++)
			{
					std::map<std::pair<uint32_t, uint32_t>, uint32_t>::iterator i = accumTX.find (std::make_pair(iterator->first.second.second, iterator->first.second.first));
					  if (i == accumTX.end ())
						{
						  accumTX[std::make_pair(iterator->first.second.second, iterator->first.second.first)] = 0;
						}
					  accumTX[std::make_pair(iterator->first.second.second, iterator->first.second.first)] += iterator->second;
			}

		// Iteration for size of received packages
		for (std::map<std::pair< uint32_t, std::pair<uint32_t, uint32_t>>,uint32_t>::const_iterator iterator = m_sizeRX.begin();
															iterator != m_sizeRX.end(); iterator++)
			{
					std::map<std::pair<uint32_t, uint32_t>, uint32_t>::iterator i = accumRX.find (std::make_pair(iterator->first.second.second, iterator->first.second.first));
					  if (i == accumRX.end ())
						{
						  accumRX[std::make_pair(iterator->first.second.second, iterator->first.second.first)] = 0;
						}
					  accumRX[std::make_pair(iterator->first.second.second, iterator->first.second.first)] += iterator->second;
			}

		// Iteration for size of received packages on time
		for (std::map<std::pair< uint32_t, std::pair<uint32_t, uint32_t>>,uint32_t>::const_iterator iterator = m_sizeRXCT.begin();
															iterator != m_sizeRXCT.end(); iterator++)
			{
					std::map<std::pair<uint32_t, uint32_t>, uint32_t>::iterator i = accumRXCT.find (std::make_pair(iterator->first.second.second, iterator->first.second.first));
					  if (i == accumRXCT.end ())
						{
						  accumRXCT[std::make_pair(iterator->first.second.second, iterator->first.second.first)] = 0;
						}
					  accumRXCT[std::make_pair(iterator->first.second.second, iterator->first.second.first)] += iterator->second;
			}

		//PDR and CF calculation
		for (std::map<std::pair<uint32_t, uint32_t>,uint32_t>::const_iterator iteratorT = counterTX.begin();
																	iteratorT != counterTX.end(); iteratorT++)
		{
				uint32_t _Node_ = iteratorT->first.first;
				uint32_t _App_ = iteratorT->first.second;

				double _PDR_ = (double) counterRX[std::make_pair(_Node_, _App_)] / (double) counterTX[std::make_pair(_Node_, _App_)];
				m_nodePDR[std::make_pair(_Node_, _App_)] = _PDR_;

				double m_currentPDR = m_nodePDR[std::make_pair(_Node_, _App_)];
				m_pdrEWMA[std::make_pair(_Node_, _App_)] = m_weightPDR * m_currentPDR + (1 - m_weightPDR) * m_pdrEWMA[std::make_pair(_Node_, _App_)];

				double _CF_ = (double) counterRXCF[std::make_pair(_Node_, _App_)] / (double) counterRX[std::make_pair(_Node_, _App_)];
				m_nodeCF[std::make_pair(_Node_, _App_)] = _CF_;

				double m_currentCF = m_nodeCF[std::make_pair(_Node_, _App_)];
				m_cfEWMA[std::make_pair(_Node_, _App_)] = m_weightCF * m_currentCF + (1 - m_weightCF) * m_cfEWMA[std::make_pair(_Node_, _App_)];

				double _BitRateTX_ = (double) ((double) accumTX[std::make_pair(_Node_, _App_)] * (double) 8 ) / (double) 1024;
				m_BitRateTX[std::make_pair(_Node_, _App_)] = _BitRateTX_;

				double _BitRateRX_ = (double) ((double) accumRX[std::make_pair(_Node_, _App_)] * (double) 8 ) / (double) 1024;
				m_BitRateRX[std::make_pair(_Node_, _App_)] = _BitRateRX_;

				double _BitRateRXCT_ = (double) ((double) accumRXCT[std::make_pair(_Node_, _App_)] * (double) 8 ) / (double) 1024;
				m_BitRateRXCT[std::make_pair(_Node_, _App_)] = _BitRateRXCT_;

				/*std::cout 	<< "**Node: " << _Node_
							<< "\t App: " << _App_
							<< "\t PDR: " << _PDR_ * 100 << "%"
							<< "\t pdrEWMA: " << m_pdrEWMA[std::make_pair(_Node_, _App_)] * 100 << "%"
							<< "\t CF: " << _CF_ * 100 << "%"
							<< "\t cfEWMA: " << m_cfEWMA[std::make_pair(_Node_, _App_)] * 100 << "%"
							<< "\t BitRate: " << _BitRate_
							<< "\n";*/
				PDRsResultsFile << Simulator::Now().GetNanoSeconds() << "\t"
					            << _Node_ << "\t"
								<< _App_ << "\t"
					            << _PDR_ << "\t"
					            << m_pdrEWMA[std::make_pair(_Node_, _App_)] << "\t"
								<< _CF_ << "\t"
					            << m_cfEWMA[std::make_pair(_Node_, _App_)] << "\t"
								<< _BitRateTX_ << "\t"
								<< _BitRateRX_ << "\t"
								<< _BitRateRXCT_ << "\t"
								<< numSeeds << "\t"
								<< mode << "\t"
								<< topology <<"\t"
								<< TrafficPattern << "\t"
								<< predictInterval << "\t"
								<< numNodes << "\n";
		}

		std::cout 	<< "PDR Total: " << (double)m_pktRX.size() / (double)m_pktTX.size()* 100 << "%"
					<< "\t CF Total: " << (double)m_timeRXCF.size() / (double)m_timeTX.size()* 100 << "%"
					<< "\t NT: " << pktNT
					<< "\n";

		//m_pktTX.clear();
		//m_pktRX.clear();
		//m_pktRXCF.clear();
		//m_sizeRX.clear();

		sendEvent=Simulator::Schedule(Seconds(1.0), &ComputePDR_and_CF , _simulationTime) ;
	}
	else{
		Simulator::Cancel (sendEvent);
	}
}

void Predict (double _simulationTime)
{
	if (Simulator::Now ().GetSeconds()<_simulationTime){ // Metodo recursivo

		// Iteration for nodes with pdr
		for (std::map<std::pair<uint32_t, uint32_t>,double>::const_iterator iterator = m_nodePDR.begin();
															iterator != m_nodePDR.end(); iterator++)
			{
			uint32_t _Node_ = iterator->first.first;
			uint32_t _App_ = iterator->first.second;

			double _cbr_ = m_rhoEWMA[_Node_];
			//double _pdrEWMA_ = m_pdrEWMA[std::make_pair(_Node_, _App_)];
			//double _cfEWMA_ = m_cfEWMA[std::make_pair(_Node_, _App_)];
			double _bitrTX_ = m_BitRateTX[std::make_pair(_Node_, _App_)];
			double _bitrRX_ = m_BitRateRX[std::make_pair(_Node_, _App_)];
			double _bitrRXCT_ = m_BitRateRXCT[std::make_pair(_Node_, _App_)];
			double _ql_ = m_queueEWMA[_Node_];
			double _pdrNode_ = m_nodePDR[std::make_pair(_Node_, _App_)];
			double _cf_ = m_nodeCF[std::make_pair(_Node_, _App_)];
			//double _nReps_ = (double) m_numReps[_Node_] / 6.0;
			double _pdrGLOB_ = (double)m_pktRX.size() / (double)m_pktTX.size();

			//m_predict[std::make_pair(_Node_, _App_)] = cc_dqn.Func(_Node_, _App_, _cbr_, _pdr_, _cf_, _bitrRX_, _ql_);
			m_predict[std::make_pair(_Node_, _App_)] = cc_dqn.Func(_Node_, _pdrGLOB_, _cbr_, _pdrNode_, _cf_, _bitrRX_, _ql_);
			//m_predict[std::make_pair(_Node_, _App_)] = 1.0;
			
			std::cout 	<< "**Node: " << _Node_
						<< "\t App: " << _App_
						<< "\t CBR: " << _cbr_ * 100 << "%"
						<< "\t PDR: " << _pdrNode_ * 100 << "%"
						<< "\t CF: " << _cf_ * 100 << "%"
						<< "\t BRTX: " << _bitrTX_
						<< "\t BRRX: " << _bitrRX_
						<< "\t BRRXCT: " << _bitrRXCT_
						<< "\t CT: " << (double) _bitrRXCT_ / (double) _bitrTX_ * 100 << "%"
						<< "\t Pred: " << m_predict[std::make_pair(_Node_, _App_)]
						<< "\n";
			}

		sendEvent=Simulator::Schedule(Seconds(predictInterval), &Predict , _simulationTime) ;
	}
	else{
		Simulator::Cancel (sendEvent);
	}
}

void ReceivePacket (Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  while (socket->Recv ())
    {
      NS_LOG_UNCOND ("Received one packet!");
    }
}

void Send_Training_Packets (Ptr<Socket> socket,  uint32_t _m_dest,  uint32_t _m_app, double _stopTime)
{

	if (Simulator::Now().GetSeconds()<_stopTime )
	{
		//m_numReps[socket->GetNode()->GetId()] = numReps;

		if (_m_app == 80) {
			
			uint32_t new_size1; // random packet size
			Ptr<ExponentialRandomVariable> _size1= CreateObject<ExponentialRandomVariable> ();
			new_size1 = _size1->GetInteger(m_packetSizeMeanApp1,1500);
			if (new_size1<12) new_size1=12;
			if (new_size1>1500) new_size1=1500;

			double  new_interval1=m_packetIntervalMeanApp1;
			Ptr<ExponentialRandomVariable> _interval1 = CreateObject<ExponentialRandomVariable> ();
			new_interval1 = _interval1->GetValue(m_packetIntervalMeanApp1,0);

			Ptr<Packet> p1 = Create<Packet> (new_size1);

			if (m_predict[std::make_pair(socket->GetNode()->GetId(), _m_app)] == 1) {
				socket->Send(p1);
				// Map variable to store transmitted packages
				m_pktTX[std::make_pair(p1->GetUid(), _m_app)] = socket->GetNode()->GetId();
				m_timeTX[std::make_pair(p1->GetUid(), std::make_pair(_m_app, socket->GetNode()->GetId()))] = Simulator::Now().GetNanoSeconds();
			}
			else {
				pktNT++;
			}

			Simulator::Schedule (Seconds(new_interval1), &Send_Training_Packets, socket, _m_dest,   _m_app, _stopTime);
		}
	}
}

std::string getLastPart(const std::string& str, char delim = '.') {
    size_t found = str.rfind(delim);
    if (found != std::string::npos)
        return str.substr(found + 1);
    else
        return str;
}

std::string extractFirstIP(const std::string& str) {
    for (size_t i = 0; i < str.length(); ++i) {
        if (isdigit(str[i])) {
            int periods = 0;
            size_t start = i;
            while (i < str.length() && (isdigit(str[i]) || str[i] == '.')) {
                if (str[i] == '.') {
                    ++periods;
                    // Check for consecutive periods or period at the end of string
                    if (i+1 == str.length() || str[i+1] == '.' || !isdigit(str[i+1])) {
                        break;
                    }
                }
                ++i;
            }
            // Check if the last parsed substring contained exactly 3 periods
            if (periods == 3) {
                return str.substr(start, i - start);
            }
        }
    }
    return "No IP address found in the data string.";
}

void
Ipv4L3Protocol_TxCallback (std::string context, Ptr< const Packet > packet, Ptr< Ipv4 > ipv4, uint32_t interface)
{
	std::ostringstream oss;
	packet->Print(oss);
	//std::cout 	<< oss.str() << "\n";

	//Skip traces with ":olsr:" in the last string
	    if (oss.str().find("ns3::olsr::PacketHeader") != std::string::npos) {
	    	return;
	    }

	// Extract the node number from the context string
	    size_t start = std::string("/NodeList/").size();
	    size_t end = context.find("/", start);
	    std::string nodeStr = context.substr(start, end - start);
	    uint32_t node = std::stoi(nodeStr);

	// Extract trace from context string
	    std::string trace = context.substr(context.length() - 2);

	// Extract ip destiny number
	    std::string data = oss.str();

		size_t found = data.find('>');
		if (found == std::string::npos) {
			std::cout << "No IP addresses found in the data string.\n";
			return;
		}

	// Find the start and end positions of the second IP address
		size_t start2 = data.find_first_not_of(' ', found + 1);
		size_t end2 = data.find_first_of(' ', start2);
		if (end2 == std::string::npos) end2 = data.size();  // in case the IP address is at the end of the string

		std::string ip2 = data.substr(start2, end2 - start2);

		std::string destinyNodeStr = getLastPart(ip2);
		uint32_t destinyNode = std::stoi(destinyNodeStr);

	// Find the start and end positions of the source IP address
		std::string ip1 = extractFirstIP(data);

		std::string sourceNodeStr = getLastPart(ip1);
		uint32_t sourceNode = std::stoi(sourceNodeStr);
	            
	// Extract port number
		std::size_t portPos = data.find_last_of(">"); // find the last '>' character
		if (portPos == std::string::npos) {
			return; // port not found, skip trace
		}
		portPos += 2; // move past the '>' and the space character that follows it
		std::size_t endPos = data.find(") Payload"); // find the end of the port number
		if (endPos == std::string::npos) {
			return; // end of port not found, skip trace
		}
		std::string portStr = data.substr(portPos, endPos - portPos); // extract port number
		uint32_t port = std::stoi(portStr);
		
	// Map variable to store transmitted packages
	   if ((trace == "Tx") && (node == sourceNode - 1)){
			   //m_pktTX[std::make_pair(packet->GetUid(), port)] = sourceNode - 1;
			   //m_timeTX[std::make_pair(packet->GetUid(), std::make_pair(port, sourceNode - 1))] = Simulator::Now().GetNanoSeconds();
		    m_sizeTX[std::make_pair(packet->GetUid(), std::make_pair(port, sourceNode - 1))] = packet->GetSize();

		   }

    // Map variable to store received packages
	   if ((trace == "Rx") && (node == destinyNode - 1)){
		    m_pktRX[std::make_pair(packet->GetUid(), port)] = sourceNode - 1;
		    m_timeRX[std::make_pair(packet->GetUid(), std::make_pair(port, sourceNode - 1))] = Simulator::Now().GetNanoSeconds();
		    m_sizeRX[std::make_pair(packet->GetUid(), std::make_pair(port, sourceNode - 1))] = packet->GetSize();
		    // Map variable to store transmitted packages on time
			for (std::map<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>,uint64_t>::const_iterator iterator = m_timeTX.begin();
														iterator != m_timeTX.end(); iterator++)
				{
						if ( iterator->first.first == packet->GetUid() &&
								m_timeRX[std::make_pair(iterator->first.first, std::make_pair(iterator->first.second.first, iterator->first.second.second))] - iterator->second <= m_topDelay) {
									
							m_pktRXCF[std::make_pair(iterator->first.first, iterator->first.second.first)] = iterator->first.second.second;
							m_timeRXCF[std::make_pair(iterator->first.first, std::make_pair(iterator->first.second.first, iterator->first.second.second))] = Simulator::Now().GetNanoSeconds();
							m_sizeRXCT[std::make_pair(iterator->first.first, std::make_pair(iterator->first.second.first, iterator->first.second.second))] = packet->GetSize();
						}
				}
	   }

	tracesResultsFile << Simulator::Now().GetNanoSeconds() << "\t"
	                    << context << "\t"
	                    << packet->GetUid() << "\t"
	                    << packet->GetSize() << "\t"
						<< oss.str() << "\t"
						<< numRun << "\t"
						<< numSeeds << "\t"
						<< sourceNode - 1 << "\t"
						<< port << "\t"
						<< m_rhoEWMA[node] << "\t"
						<< topology << "\t"
						<< mode << "\t"
						<< TrafficPattern << "\t"
						<< predictInterval << "\t"
						<< numNodes << "\n";
}

void traceSources(){
	std::ostringstream trazaText;

	trazaText.str(""); //Tx: Send ipv4 packet to outgoing interface.
	trazaText << "/NodeList/*/$ns3::Ipv4L3Protocol/Tx";
	Config::Connect (trazaText.str(), MakeCallback(&Ipv4L3Protocol_TxCallback));

	trazaText.str(""); //Tx: Send ipv4 packet to outgoing interface.
	trazaText << "/NodeList/*/$ns3::Ipv4L3Protocol/Rx";
	Config::Connect (trazaText.str(), MakeCallback(&Ipv4L3Protocol_TxCallback));
}

void installFlowMonitor(){
	
	monitor= flowmon.InstallAll();
}

void getMetricsFlowMonitor(){

		//Flow monitor
		// Define variables to calculate the metrics
		int k=0;
		int totaltxPackets = 0;
		int totalrxPackets = 0;
		double totaltxbytes = 0;
		double totalrxbytes = 0;
		double totaldelay = 0;
		double totalHopCount = 0;
		double totalrxbitrate = 0;
		double difftx, diffrx;
		double pdr_value, rxbitrate_value, txbitrate_value, delay_value, hc_value;
		double pdr_total, rxbitrate_total, delay_total, hc_total;

		//Print per flow statistics
		monitor->CheckForLostPackets ();
		Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier ());
		std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();

		int numFlows = 0 ;

		for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin ();
			  i != stats.end (); ++i)
		{
		  Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (i->first);
		  difftx = i->second.timeLastTxPacket.GetSeconds() -i->second.timeFirstTxPacket.GetSeconds();
		  diffrx = i->second.timeLastRxPacket.GetSeconds() -i->second.timeFirstRxPacket.GetSeconds();
		  pdr_value = (double) i->second.rxPackets / (double) i->second.txPackets * 100;
		  txbitrate_value = (double) i->second.txBytes * 8 / 1024 / difftx;
		  if (i->second.rxPackets != 0){
			  //rxbitrate_value = (double)i->second.rxPackets * currentMeanPacketSize * 8 /1024 / diffrx;
			  rxbitrate_value = (double)i->second.rxBytes * 8 /1024 / diffrx;
			  delay_value = (double) i->second.delaySum.GetSeconds() /(double) i->second.rxPackets;
			  hc_value = (double) i->second.timesForwarded /(double) i->second.rxPackets;
			  hc_value = hc_value+1;
		  }
		  else{
			  rxbitrate_value = 0;
			  delay_value = 0;
			  hc_value = -1000;
		  }

		  // We are only interested in the metrics of the data flows
		  if (
				  (!t.destinationAddress.IsSubnetDirectedBroadcast("255.255.255.0"))
				  && (t.destinationPort !=9999)
			  )
		  {
			  k++;// Plot the statistics for each data flow
			  if (FlowMonitorFlows){
				  std::cout << "\nFlow " << k << " (" << t.sourceAddress << " -> "<< t.destinationAddress << ")\n";
				  std::cout << "Application destination port: " << (uint16_t) t.destinationPort << "\n";
				  std::cout << "Tx Packets: " << i->second.txPackets << "\n";
				  std::cout << "Rx Packets: " << i->second.rxPackets << "\n";
				  std::cout << "Lost Packets: " << i->second.lostPackets << "\n";
				  std::cout << "Dropped Packets: " << i->second.packetsDropped.size() << "\n";
				  std::cout << "PDR:  " << pdr_value << " %\n";
				  std::cout << "Average delay: " << delay_value << "s\n";
				  std::cout << "Hop count: " << hc_value << "\n";
				  std::cout << "Rx bitrate: " <<  rxbitrate_value << " kbps\n";
				  std::cout << "Tx bitrate: " << txbitrate_value << " kbps\n\n";
			  }
			  // Acumulate for average statistics
			  totaltxPackets += i->second.txPackets;
			  totaltxbytes += i->second.txBytes;
			  totalrxPackets += i->second.rxPackets;
			  totaldelay += i->second.delaySum.GetSeconds();
			  totalHopCount += hc_value;
			  totalrxbitrate += rxbitrate_value;
			  totalrxbytes += i->second.rxBytes;
		  }

		  numFlows = numFlows +1;
		}

		// Average all nodes statistics
		if (totaltxPackets != 0){
		  pdr_total = (double) totalrxPackets / (double) totaltxPackets * 100;
		}
		else{
		  pdr_total = 0;
		}

		if (totalrxPackets != 0){
		  rxbitrate_total = totalrxbitrate;
		  delay_total = (double) totaldelay / (double) totalrxPackets;
		  hc_total = (double) totalHopCount / (double) numFlows;
		}
		else{
		  rxbitrate_total = 0;
		  delay_total = 0;
		  hc_total = -1000;
		}

		//print all nodes statistics
		if (FlowMonitorResults){
			std::cout << "\nTotal Statics: "<< "\n";
			std::cout << "Total PDR: " << pdr_total << " %\n";
			std::cout << "Total Rx bitrate: " << rxbitrate_total << " kbps\n";
			std::cout << "Total Delay: " << delay_total << " s\n";
			std::cout << "Total Hop count: " << hc_total << " \n";
			std::cout << "Total Tx Packets: " << totaltxPackets << " \n";
			std::cout << "Total Rx Packets: " << totalrxPackets << " \n";
			std::cout << "Total Rx Bytes: " << totalrxbytes << "\n\n";
		}
}

void createAPPUP (double _routTime, double _simulationTime, double _a,
		double initTimeApp, double stopTimeApp,
		uint32_t _m_source, uint32_t _nodoDestino, uint32_t numReps, uint32_t port  )
{
	double _jitterApp;
	double initTime;
	double stopTime;

	Ptr<ExponentialRandomVariable> jitterApp = CreateObject<ExponentialRandomVariable> (); 
	jitterApp->SetAttribute ("Mean", DoubleValue (0.01));jitterApp->SetAttribute ("Bound", DoubleValue (0));

	TypeId tidServerApp = TypeId::LookupByName ("ns3::UdpSocketFactory");



	for (uint32_t m_rep =  1;  m_rep <=  numReps; m_rep++){
		_jitterApp = jitterApp->GetValue ();

		initTime = _routTime + initTimeApp*a + _jitterApp;
		stopTime = _routTime + stopTimeApp*a + _jitterApp;

		InetSocketAddress remoteApp = InetSocketAddress (interfacesGlobal.GetAddress (_nodoDestino), port);
		Ptr<Socket> clientApp = Socket::CreateSocket (nodesGlobal.Get (_m_source), tidServerApp);//client
		clientApp->Connect (remoteApp); // Connect the client with the server
		Simulator::ScheduleWithContext (clientApp->GetNode ()->GetId (), Seconds (initTime), &Send_Training_Packets,
				clientApp,	_nodoDestino,  port, stopTime);
	}
}

void checkNodePositions (uint32_t numNodos, uint32_t numberOfInterfaces, std::string phyLayer)
{
	//In order to see the position and the mac address
	for (uint32_t nodPos = 0; nodPos < numNodos; nodPos++)
	{
		std::cout << "Nodo "<<nodPos+1<<": \t" <<nodesGlobal.Get(nodPos)->GetDevice(0)->GetAddress() << "\t";
		Ptr<MobilityModel> mob = nodesGlobal.Get(nodPos)->GetObject<MobilityModel> ();
		if (! mob) continue; // Strange -- node has no mobility modelinstalled. Skip.

		Vector pos = mob->GetPosition ();
		std::cout << "Position "  << " is at (" << pos.x << ", " <<
		pos.y << ", " << pos.z << "), TRANSMISSION POWER (dbm):  ";

		Ptr<NetDevice> nd = devicesGlobal.Get (nodPos);
		Ptr<WifiNetDevice> wnd = nd->GetObject<WifiNetDevice> ();

		// finally set physical parameters
		std::cout << wnd->GetMac()->GetWifiPhy()->GetTxPowerStart()
				<< ", " << wnd->GetMac()->GetWifiPhy()->GetTxPowerEnd()
				<< " Gain (db), " << wnd->GetMac()->GetWifiPhy()->GetTxGain()
				<< ", " << wnd->GetMac()->GetWifiPhy()->GetRxGain()
				<< " phyLayer: " << phyLayer

				  <<"\n" ;

	}
}

Ptr<ListPositionAllocator> load_positions (std::string device_file	)
{

	Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

	std::ifstream in(device_file); // Open for reading
	std::string s;
	while(getline(in, s))
	{ // Discards newline char
		std::string str =s;
		char delim = ',' ;
		std::stringstream ss(str);
		std::string token; // Elemento obtenido
		std::vector<std::string> cadenaNodoApp;
		while (std::getline(ss, token, delim)) {
			cadenaNodoApp.push_back(token);
		}
		//uint32_t nodeID = atoi(cadenaNodoApp[0].c_str());
		double posX = atof(cadenaNodoApp[2].c_str()) ;
		double posY = atof(cadenaNodoApp[1].c_str()) ;
		//std::string DEVICE = cadenaNodoApp[3].c_str();

  	    positionAlloc->Add (Vector (posX, posY, 0.0));

	}
	return positionAlloc;

}

int main (int argc, char *argv[])
{

	Packet::EnableChecking();
	Packet::EnablePrinting();

	bool verbose = false;
	bool tracing = false;

	double distance = 65;
	m_enableApp1 = true;
	m_portApp1 = 80;
	m_packetSizeMeanApp1 = 100;
	m_packetIntervalMeanApp1=0.05;

	double simulationTime = 48*a;
	double routTime = 30;
	
	//uint32_t aggregationFactor=0;
	
	FlowMonitorFlows = false;
	FlowMonitorResults = true;
	
	CommandLine cmd (__FILE__);
	cmd.AddValue ("TrafficPattern", "the type of traffic pattern", TrafficPattern);
	cmd.AddValue ("distance", "distance (m)", distance);
	cmd.AddValue ("verbose", "turn on all WifiNetDevice log components", verbose);
	cmd.AddValue ("tracing", "turn on ascii and pcap tracing", tracing);
	cmd.AddValue ("numNodes", "number of nodes", numNodes);
	cmd.AddValue ("numSeeds", "seed number", numSeeds);
	cmd.AddValue ("sinkNode", "Receiver node number", sinkNode);
	cmd.AddValue ("mode", "the mode of simulation", mode);
	cmd.AddValue ("topology", "The type of topology", topology);
	cmd.AddValue ("predictInterval", "The time interval to make a prediction", predictInterval);
	cmd.AddValue ("key","memory block key",memblock_key);
	cmd.Parse (argc, argv);

	RngSeedManager::SetSeed (numSeeds); 
	RngSeedManager::SetRun (numRun);   

	std::ostringstream auxFile; /*std::ostringstream cbrFile;*/ std::ostringstream pdrFile;
	auxFile.str(""); /*cbrFile.str("");*/ pdrFile.str("");

  	auxFile << "/home/pc/ns3/ns-allinone-3.35/ns-3.35/traces/"
  			<<"ADHOC_numSeeds_"<<numSeeds<<"_numRun_"<<numRun
			<<"_numNodes_"<<numNodes<<"_STime_"<<simulationTime
			<<"_Alpha_"<<m_weight<<"_pktInterval_"<<m_packetIntervalMeanApp1
			<<"_mode_"<<mode
			<<"_topology_" <<topology
			<<"_pattern_" <<TrafficPattern
			<<"_pred_" <<predictInterval
			<<".tsv";

  	/*cbrFile << "/home/pc/ns3/ns-allinone-3.35/ns-3.35/CBRs/"
  			<<"ADHOC_numSeeds_"<<numSeeds<<"_numRun_"<<numRun
			<<"_numNodes_"<<numNodes<<"_STime_"<<simulationTime
			<<"_Alpha_"<<m_weight<<"_pktInterval_"<<m_packetIntervalMeanApp1
			<<"_mode_"<<mode
			<<"_topology_" <<topology
			<<"_pattern_" <<TrafficPattern
			<<"_pred_" <<predictInterval
			<<".tsv";*/

  	pdrFile << "/home/pc/ns3/ns-allinone-3.35/ns-3.35/PDRs/"
  	  		<<"ADHOC_numSeeds_"<<numSeeds<<"_numRun_"<<numRun
  			<<"_numNodes_"<<numNodes<<"_STime_"<<simulationTime
  			<<"_Alpha_"<<m_weight<<"_pktInterval_"<<m_packetIntervalMeanApp1
  			<<"_mode_"<<mode
  			<<"_topology_" <<topology
			<<"_pattern_" <<TrafficPattern
			<<"_pred_" <<predictInterval
  			<<".tsv";

  	tracesResultsFile.open (auxFile.str().c_str());
  	//CBRsResultsFile.open (cbrFile.str().c_str());
  	PDRsResultsFile.open (pdrFile.str().c_str());

	nodesGlobal.Create (numNodes);
	
	// The below set of helpers will help us to put together the wifi NICs we want
	WifiHelper wifi;
	if (verbose)
	{
		wifi.EnableLogComponents ();  // Turn on all Wifi logging
	}
	
	YansWifiPhyHelper wifiPhy;
	// set it to zero; otherwise, gain will be added
	wifiPhy.Set ("RxGain", DoubleValue (-10) );
	// ns-3 supports RadioTap and Prism tracing extensions for 802.11b
	wifiPhy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);

	YansWifiChannelHelper wifiChannel;
	wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
	wifiChannel.AddPropagationLoss ("ns3::FriisPropagationLossModel");
	wifiPhy.SetChannel (wifiChannel.Create ());

	// Add an upper mac and disable rate control
	std::string phyMode ("DsssRate5_5Mbps");
	// Fix non-unicast data rate to be the same as that of unicast
	Config::SetDefault ("ns3::WifiRemoteStationManager::NonUnicastMode",
					  StringValue (phyMode));
	WifiMacHelper wifiMac;
	wifi.SetStandard (WIFI_STANDARD_80211b);
	wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
								"DataMode",StringValue (phyMode),
								"ControlMode",StringValue (phyMode));
	// Set it to adhoc mode
	wifiMac.SetType ("ns3::AdhocWifiMac");
	devicesGlobal= wifi.Install (wifiPhy, wifiMac, nodesGlobal);
	
	MobilityHelper mobility;
	Ptr<PositionAllocator> taPositionAlloc;
	ObjectFactory pos;
	
	if (topology == "SquareGrid") {
		mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
										"MinX", DoubleValue (0.0),
										"MinY", DoubleValue (0.0),
										"DeltaX", DoubleValue (distance),
										"DeltaY", DoubleValue (distance),
										"GridWidth", UintegerValue (floor(sqrt(numNodes))),
										"LayoutType", StringValue ("RowFirst"));
		mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
		mobility.Install (nodesGlobal);
	}

	if (topology == "Montreal") {
		if (numNodes == 9) {
			std::string deviceINFORMATION_file = "/home/pc/ns3/ns-allinone-3.35/ns-3.35/Montreal/POSITIONS_MONTREAL_8.csv";
			mobility.SetPositionAllocator (load_positions (deviceINFORMATION_file	));
		}
		if (numNodes == 16) {
			std::string deviceINFORMATION_file = "/home/pc/ns3/ns-allinone-3.35/ns-3.35/Montreal/POSITIONS_MONTREAL_15.csv";
			mobility.SetPositionAllocator (load_positions (deviceINFORMATION_file	));
		}
		if (numNodes == 25) {
			std::string deviceINFORMATION_file = "/home/pc/ns3/ns-allinone-3.35/ns-3.35/Montreal/POSITIONS_MONTREAL_24.csv";
			mobility.SetPositionAllocator (load_positions (deviceINFORMATION_file	));
		}
		mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
		mobility.Install (nodesGlobal);
	}

	if (topology == "Mobile") {
		int64_t streamIndex = 0;
		double nodeSpeed = 3.5;//[m/s]
		double nodePause = 0;//[s]
		ObjectFactory pos;
		pos.SetTypeId ("ns3::RandomRectanglePositionAllocator");
		pos.Set ("X", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=200.0]"));
		pos.Set ("Y", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=200.0]"));

		Ptr<PositionAllocator> taPositionAlloc = pos.Create ()->GetObject<PositionAllocator> ();
		streamIndex += taPositionAlloc->AssignStreams (streamIndex);

		std::stringstream ssSpeed;
		ssSpeed << "ns3::UniformRandomVariable[Min=0.0|Max=" << nodeSpeed << "]";
		std::stringstream ssPause;
		ssPause << "ns3::ConstantRandomVariable[Constant=" << nodePause << "]";
		mobility.SetMobilityModel ("ns3::RandomWaypointMobilityModel",
									  "Speed", StringValue (ssSpeed.str ()),
									  "Pause", StringValue (ssPause.str ()),
									  "PositionAllocator", PointerValue (taPositionAlloc));
		mobility.SetPositionAllocator (taPositionAlloc);
		mobility.Install (nodesGlobal);
		streamIndex += mobility.AssignStreams (nodesGlobal, streamIndex);

		/*std::string netAnimFile = "/home/pc/ns3/ns-allinone-3.35/ns-3.35/mobilityTrace/" ;
		netAnimFile <<"ADHOC_numSeeds_"<<numSeeds<<"_numRun_"<<numRun
					<<"_numNodes_"<<numNodes<<"_STime_"<<simulationTime
					<<"_Alpha_"<<m_weight<<"_pktInterval_"<<m_packetIntervalMeanApp1
					<<"_mode_"<<mode
					<<"_topology_" <<topology
					<<"_pattern_" <<TrafficPattern
					<<"_pred_" <<predictInterval
					<<".xml";
		AnimationInterface anim (netAnimFile);*/
	}


	// Enable OLSR
	OlsrHelper olsr;
	Ipv4StaticRoutingHelper staticRouting;
	
	Ipv4ListRoutingHelper list;
	list.Add (staticRouting, 0);
	list.Add (olsr, 10);
	
	InternetStackHelper internet;
	internet.SetRoutingHelper (list); // has effect on the next Install ()
	internet.Install (nodesGlobal);
	
	Ipv4AddressHelper ipv4;
	NS_LOG_INFO ("Assign IP Addresses.");
	ipv4.SetBase ("10.1.1.0", "255.255.255.0");
	interfacesGlobal = ipv4.Assign (devicesGlobal);

	checkNodePositions (numNodes, 1, "802.11b");

	// ========CREATE SERVER SOCKET ON SINK NODE
	TypeId tidServerApp1 = TypeId::LookupByName ("ns3::UdpSocketFactory");

	if (m_enableApp1){
	  Ptr<Socket> recvSinkApp1 = Socket::CreateSocket (nodesGlobal.Get (sinkNode), tidServerApp1);//Destination
	  InetSocketAddress local1 = InetSocketAddress (interfacesGlobal.GetAddress (sinkNode), m_portApp1);
	  recvSinkApp1->Bind (local1);
	  recvSinkApp1->SetRecvCallback (MakeCallback (&ReceivePacket));
	}
    //std::srand(numSeeds); // seed random number generator
	std::vector<int> choices { 1, 2, 3, 4, 5 };

	// Create applications on all nodes, sending packets to the sink node
	for (uint32_t s_d = 0; s_d < numNodes; s_d++)
	{
		if (s_d != sinkNode)
		{
			m_predict[std::make_pair(s_d, m_portApp1)] = 1.0;
			//m_numReps[s_d] = 0;
			
			if (TrafficPattern == "Random") {
				for (double t = 0.0; t < 32.0; t += 1.0) {
					std::srand(s_d + numSeeds + t); // seed random number generator with unique seed for each s_d and t
					int index = std::rand() % choices.size();
					createAPPUP (routTime, simulationTime, a, t, t+1.0, s_d, sinkNode, choices[index], m_portApp1);
				}
			}

			if (TrafficPattern == "Constant") {
				for (double t = 0.0; t < 32.0; t += 1.0) {
					createAPPUP (routTime, simulationTime, a, t, t+1.0, s_d, sinkNode, 5, m_portApp1);
				}
			}

			else if (TrafficPattern == "RollerCoaster") {
				createAPPUP (routTime, simulationTime, a,  0.0,  1.0, s_d, sinkNode, 5, m_portApp1 );
				createAPPUP (routTime, simulationTime, a,  1.0,  2.0, s_d, sinkNode, 4, m_portApp1 );
				createAPPUP (routTime, simulationTime, a,  2.0,  3.0, s_d, sinkNode, 3, m_portApp1 );
				createAPPUP (routTime, simulationTime, a,  3.0,  4.0, s_d, sinkNode, 4, m_portApp1 );
				createAPPUP (routTime, simulationTime, a,  4.0,  5.0, s_d, sinkNode, 2, m_portApp1 );
				createAPPUP (routTime, simulationTime, a,  5.0,  6.0, s_d, sinkNode, 3, m_portApp1 );
				createAPPUP (routTime, simulationTime, a,  6.0,  7.0, s_d, sinkNode, 1, m_portApp1 );
				createAPPUP (routTime, simulationTime, a,  7.0,  8.0, s_d, sinkNode, 2, m_portApp1 );
				createAPPUP (routTime, simulationTime, a,  8.0,  9.0, s_d, sinkNode, 2, m_portApp1 );
				createAPPUP (routTime, simulationTime, a,  9.0, 10.0, s_d, sinkNode, 1, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 10.0, 11.0, s_d, sinkNode, 3, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 11.0, 12.0, s_d, sinkNode, 2, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 12.0, 13.0, s_d, sinkNode, 4, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 13.0, 14.0, s_d, sinkNode, 3, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 14.0, 15.0, s_d, sinkNode, 4, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 15.0, 16.0, s_d, sinkNode, 5, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 16.0, 17.0, s_d, sinkNode, 5, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 17.0, 18.0, s_d, sinkNode, 4, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 18.0, 19.0, s_d, sinkNode, 3, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 19.0, 20.0, s_d, sinkNode, 4, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 20.0, 21.0, s_d, sinkNode, 2, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 21.0, 22.0, s_d, sinkNode, 3, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 22.0, 23.0, s_d, sinkNode, 1, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 23.0, 24.0, s_d, sinkNode, 2, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 24.0, 25.0, s_d, sinkNode, 2, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 25.0, 26.0, s_d, sinkNode, 1, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 26.0, 27.0, s_d, sinkNode, 3, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 27.0, 28.0, s_d, sinkNode, 2, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 28.0, 29.0, s_d, sinkNode, 4, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 29.0, 30.0, s_d, sinkNode, 3, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 30.0, 31.0, s_d, sinkNode, 4, m_portApp1 );
				createAPPUP (routTime, simulationTime, a, 31.0, 32.0, s_d, sinkNode, 5, m_portApp1 );
			}
		}
	}

	traceSources();
	installFlowMonitor();
	
	//===to trigger the channel utilization measurement
	for (uint32_t idNodo = 0; idNodo < numNodes; ++idNodo)
	{
		sendEvent=Simulator::Schedule(Seconds(1.01+idNodo*0.01), &CBRSample ,
				idNodo, nodesGlobal, devicesGlobal, simulationTime) ;
	}
	
	sendEvent=Simulator::Schedule(Seconds(routTime), &ComputePDR_and_CF ,simulationTime) ;	
	sendEvent=Simulator::Schedule(Seconds(routTime), &Predict ,simulationTime) ;
	
	Simulator::Stop (Seconds (simulationTime));
	Simulator::Run ();

	getMetricsFlowMonitor();

	cc_dqn.SetFinish();
	
	Simulator::Destroy ();
	tracesResultsFile.close();
	//CBRsResultsFile.close();
	PDRsResultsFile.close();
	
	return 0;
}