package fr.unice.polytech.si5.rimel.eventprofile.model;

import lombok.Builder;
import org.neo4j.ogm.annotation.*;

@RelationshipEntity
@Builder
public class HasProfile {

	@Id
	@GeneratedValue
	private Long id;

	@Builder.Default
	@Property
	private boolean isDefault = false;

	@StartNode
	private Pom pom;

	@EndNode
	private MavenProfile mavenProfile;

	public HasProfile(Long id, boolean isDefault, Pom pom, MavenProfile mavenProfile) {
		this.id = id;
		this.isDefault = isDefault;
		this.pom = pom;
		this.mavenProfile = mavenProfile;
	}

	public HasProfile() {
	}

	public Long getId() {
		return id;
	}

	public void setId(Long id) {
		this.id = id;
	}

	public boolean isDefault() {
		return isDefault;
	}

	public void setDefault(boolean aDefault) {
		isDefault = aDefault;
	}

	public Pom getPom() {
		return pom;
	}

	public void setPom(Pom pom) {
		this.pom = pom;
	}

	public MavenProfile getMavenProfile() {
		return mavenProfile;
	}

	public void setMavenProfile(MavenProfile mavenProfile) {
		this.mavenProfile = mavenProfile;
	}
}
