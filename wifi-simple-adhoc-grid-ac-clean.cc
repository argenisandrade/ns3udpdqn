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
#include "ns3/ns3-ai-module.h"

using namespace std;
using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("wifi-simple-adhoc-grid-ac");

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

class APB : public Ns3AIRL<Env, Act>
{
public:
    APB(uint16_t id);
    int Func(double node, double app, double cbr, double pdr, double cf, double bitrate, double ql);
};

APB::APB(uint16_t id) : Ns3AIRL<Env, Act>(id) {
    SetCond(2, 0);      ///< Set the operation lock (even for ns-3 and odd for python).
}

int APB::Func(double node, double app, double cbr, double pdr, double cf, double bitrate, double ql)
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

int memblock_key = 2333;        ///< memory block key, need to keep the same in the python script

double c = 1.0;			//Prediction
APB apb(memblock_key);

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
uint32_t numNodes = 3;
uint32_t numSeeds = 4;
uint32_t numRun = 5;
double mode;
std::string topology = "linear";

std::ofstream tracesResultsFile; 
std::ofstream CBRsResultsFile;
std::ofstream PDRsResultsFile;

double measureMean=0.001;
EventId sendEvent;
double m_weight = 0.0005;
double m_weightPDR = 0.05;
double m_weightCF = 0.05;
uint64_t m_topDelay = 5000000;
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

std::map<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>,uint32_t> m_sizeRX;
std::map<std::pair<uint32_t, uint32_t>,double> m_BitRate;

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

	CBRsResultsFile << Simulator::Now().GetNanoSeconds() << "\t"
			<< node << "\t"
			<< m_rhoEWMA[node]<< "\t"
			<< m_queueEWMA[node] << "\t"
			<< m_currentRho<< "\t"
			<< currentQP << "\t"
			<< numSeeds << "\t"
			<< mode << "\t"
			<< topology <<"\n";

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
	std::map<std::pair<uint32_t, uint32_t>,uint32_t> accumRX;

	if (Simulator::Now ().GetSeconds()<_simulationTime){ // Metodo recursivo

		std::cout << "\n"<< Simulator::Now ().GetMilliSeconds() << " Seed: " << numSeeds << std::endl;

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

				double _CF_ = (double) counterRXCF[std::make_pair(_Node_, _App_)] / (double) counterTX[std::make_pair(_Node_, _App_)];
				m_nodeCF[std::make_pair(_Node_, _App_)] = _CF_;

				double m_currentCF = m_nodeCF[std::make_pair(_Node_, _App_)];
				m_cfEWMA[std::make_pair(_Node_, _App_)] = m_weightCF * m_currentCF + (1 - m_weightCF) * m_cfEWMA[std::make_pair(_Node_, _App_)];

				double _BitRate_ = (double) ((double) accumRX[std::make_pair(_Node_, _App_)] * (double) 8 ) / (double) 1024;
				m_BitRate[std::make_pair(_Node_, _App_)] = _BitRate_;

				std::cout 	<< "**Node: " << _Node_
							<< "\t App: " << _App_
							<< "\t PDR: " << _PDR_ * 100 << "%"
							<< "\t pdrEWMA: " << m_pdrEWMA[std::make_pair(_Node_, _App_)] * 100 << "%"
							<< "\t CF: " << _CF_ * 100 << "%"
							<< "\t cfEWMA: " << m_cfEWMA[std::make_pair(_Node_, _App_)] * 100 << "%"
							<< "\t BitRate: " << _BitRate_
							<< "\n";
				PDRsResultsFile << Simulator::Now().GetNanoSeconds() << "\t"
					            << _Node_ << "\t"
								<< _App_ << "\t"
					            << _PDR_ << "\t"
					            << m_pdrEWMA[std::make_pair(_Node_, _App_)] << "\t"
								<< _CF_ << "\t"
					            << m_cfEWMA[std::make_pair(_Node_, _App_)] << "\t"
								<< _BitRate_ << "\t"
								<< numSeeds << "\t"
								<< mode << "\t"
								<< topology <<"\n";
		}

		std::cout 	<< "PDR Total: " << (double)m_timeRX.size() / (double)m_timeTX.size()* 100 << "%"
					<< "\t CF Total: " << (double)m_timeRXCF.size() / (double)m_timeTX.size()* 100 << "%"
					<< "\t NT: " << pktNT
					<< "\n";

		//m_pktTX.clear();
		//m_pktRX.clear();
		//m_pktRXCF.clear();
		m_sizeRX.clear();

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
		for (std::map<std::pair<uint32_t, uint32_t>,double>::const_iterator iterator = m_pdrEWMA.begin();
															iterator != m_pdrEWMA.end(); iterator++)
			{
			uint32_t _Node_ = iterator->first.first;
			uint32_t _App_ = iterator->first.second;

			double _cbr_ = m_rhoEWMA[_Node_];
			double _pdr_ = m_pdrEWMA[std::make_pair(_Node_, _App_)];
			double _cf_ = m_cfEWMA[std::make_pair(_Node_, _App_)];
			double _bitr_ = m_BitRate[std::make_pair(_Node_, _App_)];
			double _ql_ = m_queueEWMA[_Node_];
			//double _pdr_ = m_nodePDR[std::make_pair(_Node_, _App_)];
			//double _cf_ = m_nodeCF[std::make_pair(_Node_, _App_)];

			m_predict[std::make_pair(_Node_, _App_)] = apb.Func(_Node_, _App_, _cbr_, _pdr_, _cf_, _bitr_, _ql_);

			std::cout 	<< "Node: " << _Node_
						<< "\t App: " << _App_
						<< "\t CBR: " << _cbr_ * 100 << "%"
						<< "\t PDR: " << _pdr_ * 100 << "%"
						<< "\t CF: " << _cf_ * 100 << "%"
						<< "\t Pred: " << m_predict[std::make_pair(_Node_, _App_)]
						<< "\n";
			}

		sendEvent=Simulator::Schedule(Seconds(1.0), &Predict , _simulationTime) ;
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
		if (m_pdrEWMA.size() == 0)
		{
			c = 1.0;
		}
		else
		{
			c = m_predict[std::make_pair(socket->GetNode()->GetId(), _m_app)];
		}
		
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

			if (c == 1) {
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


		if (_m_app == 81) {
				
			uint32_t new_size1; // random packet size
			Ptr<ExponentialRandomVariable> _size1= CreateObject<ExponentialRandomVariable> ();
			new_size1 = _size1->GetInteger(m_packetSizeMeanApp2,1500);
			if (new_size1<12) new_size1=12;
			if (new_size1>1500) new_size1=1500;

			double  new_interval1=m_packetIntervalMeanApp2;
			Ptr<ExponentialRandomVariable> _interval1 = CreateObject<ExponentialRandomVariable> ();
			new_interval1 = _interval1->GetValue(m_packetIntervalMeanApp2,0);

		    Ptr<Packet> p1 = Create<Packet> (new_size1);

		    if (c == 1) {
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
						<< mode << "\n";
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

int main (int argc, char *argv[])
{

	Packet::EnableChecking();
	Packet::EnablePrinting();

	double distance = 100;  
	uint32_t sinkNode = 0;
	bool verbose = false;
	bool tracing = false;
	
	m_enableApp1 = true;
	m_portApp1 = 80;
	m_packetSizeMeanApp1 = 100;
	m_packetIntervalMeanApp1=0.01;
	
	m_enableApp2 = true;
    m_portApp2 = 81;
    m_packetSizeMeanApp2 = 100;
    m_packetIntervalMeanApp2=0.01;

	double simulationTime = 32*a;
	double routTime = 30;
	
	uint32_t aggregationFactor=0;
	
	FlowMonitorFlows = false;
	FlowMonitorResults = true;
	
	CommandLine cmd (__FILE__);
	cmd.AddValue ("distance", "distance (m)", distance);
	cmd.AddValue ("verbose", "turn on all WifiNetDevice log components", verbose);
	cmd.AddValue ("tracing", "turn on ascii and pcap tracing", tracing);
	cmd.AddValue ("numNodes", "number of nodes", numNodes);
	cmd.AddValue ("numSeeds", "seed number", numSeeds);
	cmd.AddValue ("sinkNode", "Receiver node number", sinkNode);
	cmd.AddValue ("mode", "the mode of simulation", mode);
	cmd.AddValue ("topology", "The type of topology", topology);
	cmd.AddValue ("key","memory block key",memblock_key);
	cmd.Parse (argc, argv);

	RngSeedManager::SetSeed (numSeeds); 
	RngSeedManager::SetRun (numRun);   

	std::ostringstream auxFile; std::ostringstream cbrFile; std::ostringstream pdrFile;
	auxFile.str(""); cbrFile.str(""); pdrFile.str("");

  	auxFile << "/home/pc/ns3/ns-allinone-3.35/ns-3.35/traces/"
  			<<"ADHOC_numSeeds_"<<numSeeds<<"_numRun_"<<numRun
			<<"_numNodes_"<<numNodes<<"_STime_"<<simulationTime
			<<"_Alpha_"<<m_weight<<"_pktInterval_"<<m_packetIntervalMeanApp1
			<<"_mode_"<<mode
			<<"_topology_" <<topology
			<<".tsv";

  	cbrFile << "/home/pc/ns3/ns-allinone-3.35/ns-3.35/CBRs/"
  			<<"ADHOC_numSeeds"<<numSeeds<<"numRun"<<numRun
			<<"numNodes"<<numNodes<<"STime"<<simulationTime
			<<"Alpha"<<m_weight<<"pktInterval"<<m_packetIntervalMeanApp1
			<<"_mode_"<<mode
			<<"_topology_" <<topology
			<<".tsv";

  	pdrFile << "/home/pc/ns3/ns-allinone-3.35/ns-3.35/PDRs/"
  	  		<<"ADHOC_numSeeds"<<numSeeds<<"numRun"<<numRun
  			<<"numNodes"<<numNodes<<"STime"<<simulationTime
  			<<"Alpha"<<m_weight<<"pktInterval"<<m_packetIntervalMeanApp1
  			<<"_mode_"<<mode
  			<<"_topology_" <<topology
  			<<".tsv";

  	tracesResultsFile.open (auxFile.str().c_str());
  	CBRsResultsFile.open (cbrFile.str().c_str());
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
	
	YansWifiChannelHelper wifiChannel;
	wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
	wifiChannel.AddPropagationLoss ("ns3::FriisPropagationLossModel");
	wifiPhy.SetChannel (wifiChannel.Create ());

	// Add an upper mac and disable rate control
	WifiMacHelper wifiMac;
	wifi.SetStandard (WIFI_STANDARD_80211ac);
	std::ostringstream oss;
	//oss << "VhtMcs" << mcs;
	oss << "VhtMcs" << "0";
	wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager","DataMode", StringValue (oss.str ()),
				"ControlMode", StringValue (oss.str ()));
	
	// Set it to adhoc mode
	wifiMac.SetType("ns3::AdhocWifiMac",
					"BE_MaxAmpduSize", UintegerValue (aggregationFactor*(0+72)),
					"BE_MaxAmsduSize", UintegerValue (0)
								);
	devicesGlobal= wifi.Install (wifiPhy, wifiMac, nodesGlobal);
	
	MobilityHelper mobility;
	
	if (topology == "linear") {
	mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
									"MinX", DoubleValue (0.0),
									"MinY", DoubleValue (0.0),
									"DeltaX", DoubleValue (distance),
									"DeltaY", DoubleValue (distance),
									"GridWidth", UintegerValue (5),
									"LayoutType", StringValue ("RowFirst"));
	}
	
	mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
	mobility.Install (nodesGlobal);

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

	// ========CREATE SERVER SOCKET ON SINK NODE
	TypeId tidServerApp1 = TypeId::LookupByName ("ns3::UdpSocketFactory");

	if (m_enableApp1){
	  Ptr<Socket> recvSinkApp1 = Socket::CreateSocket (nodesGlobal.Get (sinkNode), tidServerApp1);//Destination
	  InetSocketAddress local1 = InetSocketAddress (interfacesGlobal.GetAddress (sinkNode), m_portApp1);
	  recvSinkApp1->Bind (local1);
	  recvSinkApp1->SetRecvCallback (MakeCallback (&ReceivePacket));
	}
	if (m_enableApp2){
	  Ptr<Socket> recvSinkApp1 = Socket::CreateSocket (nodesGlobal.Get (sinkNode), tidServerApp1);//Destination
	  InetSocketAddress local1 = InetSocketAddress (interfacesGlobal.GetAddress (sinkNode), m_portApp2);
	  recvSinkApp1->Bind (local1);
	  recvSinkApp1->SetRecvCallback (MakeCallback (&ReceivePacket));
	}
	// Create applications on all nodes, sending packets to the sink node
	for (uint32_t s_d = 0; s_d < numNodes; s_d++)
	{
	    if (s_d != sinkNode)
	    {
	  	m_predict[std::make_pair(s_d, m_portApp1)] = 1.0;
	  	m_predict[std::make_pair(s_d, m_portApp2)] = 1.0;
	    createAPPUP (routTime, simulationTime, a,  0.0,  1.0, s_d, sinkNode, 20, m_portApp1 );
	    createAPPUP (routTime, simulationTime, a,  1.0,  2.0, s_d, sinkNode, 10, m_portApp2 );
	    createAPPUP (routTime, simulationTime, a,  2.0,  3.0, s_d, sinkNode, 30, m_portApp1 );
		createAPPUP (routTime, simulationTime, a,  3.0,  4.0, s_d, sinkNode, 20, m_portApp2 );
		createAPPUP (routTime, simulationTime, a,  4.0,  5.0, s_d, sinkNode, 40, m_portApp1 );
		createAPPUP (routTime, simulationTime, a,  5.0,  6.0, s_d, sinkNode, 30, m_portApp2 );
		createAPPUP (routTime, simulationTime, a,  6.0,  7.0, s_d, sinkNode, 40, m_portApp1 );
		createAPPUP (routTime, simulationTime, a,  7.0,  8.0, s_d, sinkNode, 50, m_portApp2 );
		createAPPUP (routTime, simulationTime, a,  8.0,  9.0, s_d, sinkNode, 50, m_portApp1 );
		createAPPUP (routTime, simulationTime, a,  9.0, 10.0, s_d, sinkNode, 40, m_portApp2 );
		createAPPUP (routTime, simulationTime, a, 10.0, 11.0, s_d, sinkNode, 30, m_portApp1 );
		createAPPUP (routTime, simulationTime, a, 11.0, 12.0, s_d, sinkNode, 40, m_portApp2 );
		createAPPUP (routTime, simulationTime, a, 12.0, 13.0, s_d, sinkNode, 20, m_portApp1 );
		createAPPUP (routTime, simulationTime, a, 13.0, 14.0, s_d, sinkNode, 30, m_portApp2 );
		createAPPUP (routTime, simulationTime, a, 14.0, 15.0, s_d, sinkNode, 10, m_portApp1 );
		createAPPUP (routTime, simulationTime, a, 15.0, 16.0, s_d, sinkNode, 20, m_portApp2 );
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

	apb.SetFinish();
	
	Simulator::Destroy ();
	tracesResultsFile.close();
	CBRsResultsFile.close();
	PDRsResultsFile.close();
	
	return 0;
}