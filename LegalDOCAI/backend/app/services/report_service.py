import os
import json
import datetime
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

class ReportGenerationService:
    def __init__(self, output_dir: str = "d:\\Legal-mohan\\Legaldoc-new\\LegalDOCAI\\outputs\\reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='CenterTitle',
            parent=self.styles['Title'],
            alignment=TA_CENTER,
            fontSize=24,
            spaceAfter=30
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            color=colors.HexColor("#2C3E50"),
            spaceBefore=20,
            spaceAfter=10,
            borderPadding=5,
            borderWidth=0,
            leftIndent=0
        ))
        self.styles.add(ParagraphStyle(
            name='NormalLeft',
            parent=self.styles['Normal'],
            alignment=TA_LEFT,
            fontSize=10,
            leading=14
        ))

    def create_risk_chart(self, risk_data: Dict[str, int]) -> str:
        """
        Creates a risk distribution pie chart using Plotly and saves as PNG.
        """
        labels = ['High Risk', 'Medium Risk', 'Low Risk']
        values = [risk_data.get('high_risk_clauses', 0), 
                  risk_data.get('medium_risk_clauses', 0), 
                  risk_data.get('low_risk_clauses', 0)]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3,
                                     marker_colors=['#E74C3C', '#F39C12', '#27AE60'])])
        fig.update_layout(title_text="Risk Distribution", width=400, height=400)
        
        img_path = os.path.join(self.output_dir, f"risk_chart_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        fig.write_image(img_path)
        return img_path

    def create_confidence_chart(self, metrics: Dict[str, float]) -> str:
        """
        Creates a confidence bar chart using Plotly.
        """
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        fig = go.Figure([go.Bar(x=labels, y=values, marker_color='#3498DB')])
        fig.update_layout(title_text="Module Confidence Scores", yaxis_range=[0, 100], width=500, height=300)
        
        img_path = os.path.join(self.output_dir, f"conf_chart_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        fig.write_image(img_path)
        return img_path

    def generate_pdf_report(self, data: Dict[str, Any]) -> str:
        """
        Main pipeline for Module 13: Professional PDF Report Generation.
        """
        file_name = data.get("document_info", {}).get("file_name", "Document")
        report_path = os.path.join(self.output_dir, f"Analysis_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        
        doc = SimpleDocTemplate(report_path, pagesize=A4)
        story = []

        # --- COVER PAGE ---
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("Legal Document Analysis Report", self.styles['CenterTitle']))
        story.append(Paragraph(f"Document: {file_name}", self.styles['Heading2']))
        story.append(Paragraph(f"Date Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        story.append(Spacer(1, 4*inch))
        story.append(Paragraph("Confidential Regulatory Compliance Report", self.styles['Normal']))
        story.append(PageBreak())

        # --- EXECUTIVE SUMMARY ---
        story.append(Paragraph("1. Executive Summary", self.styles['SectionHeader']))
        story.append(Paragraph(data.get("summary", "No summary available."), self.styles['NormalLeft']))
        story.append(Spacer(1, 12))

        # --- DOCUMENT INFO ---
        story.append(Paragraph("2. Document Information", self.styles['SectionHeader']))
        info = data.get("document_info", {})
        doc_info_data = [
            ["Attribute", "Value"],
            ["File Name", info.get("file_name", "N/A")],
            ["Language", info.get("language", "English")],
            ["Type", data.get("classification", {}).get("document_type", "N/A")],
            ["Authenticity", f"{data.get('authenticity', {}).get('label', 'Unknown')} ({data.get('authenticity', {}).get('confidence', 0)*100:.1f}%)"]
        ]
        t = Table(doc_info_data, colWidths=[2*inch, 3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#D5D8DC")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(t)

        # --- RISK ANALYSIS ---
        story.append(Paragraph("3. Risk Analysis", self.styles['SectionHeader']))
        risk = data.get("risk_analysis", {})
        story.append(Paragraph(f"Overall Risk Level: <b>{risk.get('overall_risk', 'Unknown')}</b>", self.styles['Normal']))
        
        # Risk Chart
        try:
            risk_img = self.create_risk_chart(risk)
            story.append(Image(risk_img, width=3*inch, height=3*inch))
        except Exception as e:
            story.append(Paragraph(f"Chart error: {e}", self.styles['Normal']))

        # --- COMPLIANCE ---
        story.append(Paragraph("4. Regulatory Compliance", self.styles['SectionHeader']))
        comp = data.get("compliance", {})
        story.append(Paragraph(f"Status: {comp.get('status', 'Unknown')}", self.styles['Normal']))
        violations = comp.get("violations", [])
        if violations:
            story.append(Paragraph("Violations Detected:", self.styles['Normal']))
            for v in violations:
                story.append(Paragraph(f"• {v}", self.styles['Normal']))
        else:
            story.append(Paragraph("No major regulatory violations detected.", self.styles['Normal']))

        # --- ENTITIES ---
        story.append(Paragraph("5. Extracted Entities", self.styles['SectionHeader']))
        ents = data.get("entities", {})
        ent_data = [["Category", "Entities"]]
        for cat in ["names", "dates", "amounts", "locations"]:
            items = ents.get(cat, [])
            ent_data.append([cat.capitalize(), ", ".join(items[:10]) if items else "None detected"])
        
        t_ent = Table(ent_data, colWidths=[1.5*inch, 4*inch])
        t_ent.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(t_ent)

        # --- TIMELINE ---
        if data.get("timeline"):
            story.append(PageBreak())
            story.append(Paragraph("6. Chronological Timeline", self.styles['SectionHeader']))
            timeline_data = [["Date", "Event"]]
            for item in data.get("timeline", [])[:15]:
                timeline_data.append([item.get("date", ""), item.get("event", "")])
            
            t_time = Table(timeline_data, colWidths=[1.2*inch, 4.3*inch])
            t_time.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#D5D8DC")),
            ]))
            story.append(t_time)

        # Build PDF
        doc.build(story)
        return report_path

report_service = ReportGenerationService()
